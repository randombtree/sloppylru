/// SloppyLRU cache
/// @todo Use plain file + log for rmap transaction for faster insert speed
use std::str;
use std::sync::{Arc, RwLock, Weak};
use futures::channel::{mpsc, oneshot};
use futures::future::Future;
use futures::stream::StreamExt;
use log::{
    debug,
    error,
    trace,
    warn,
};

use sled;
use sled::Transactional;

use crate::result::{
    Error,
    Result,
};
use crate::config::CacheConfig;
use crate::manager::CacheManager;
use crate::lru::{LruArray, Slot, LRU_PAGE_SIZE};
use crate::key::Key;

type Inner<const N: usize, const K: usize> = CacheInner<N, K>;


/// A LRU cache with N levels
pub struct Cache<const N: usize, const K: usize>(Arc<Inner<N, K>>)
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized;


/// Weak cache reference for manager to use, with the matching TryInto:s
pub(crate) struct WeakCache<const N: usize, const K: usize>(Weak<Inner<N, K>>)
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized;

impl<const N: usize, const K: usize> WeakCache<N,K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    pub fn new(cache: &Cache<N,K>) -> WeakCache<N,K> {
	WeakCache(Arc::downgrade(&cache.0))
    }
}

impl<const N: usize, const K: usize> TryInto<Cache<N,K>> for WeakCache<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    type Error = ();
    fn try_into(self) -> std::result::Result<Cache<N,K>, Self::Error> {
	match self.0.upgrade() {
	    Some(arc) => Ok(Cache(arc)),
	    _ => Err(()),
	}
    }
}

impl<const N: usize, const K: usize> TryInto<Cache<N,K>> for &WeakCache<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized {
    type Error = ();
    fn try_into(self) -> std::result::Result<Cache<N,K>, Self::Error> {
	match self.0.upgrade() {
	    Some(arc) => Ok(Cache(arc)),
	    _ => Err(()),
	}
    }
}

pub (crate) enum FlusherMsg {
    /// Collect old entries to make room for new ones in LRU
    GC,
    /// Emergency GC, sender is waiting for free space
    GcSync(oneshot::Sender<()>),
    /// Shut down flusher
    SHUTDOWN,
}

/// Async flusher of Cache; Runs without references
async fn flusher<const N: usize, const K: usize, F>(this: Weak<CacheInner<N, K>>, mut receiver: mpsc::Receiver<FlusherMsg>, mut purger: F)
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized,
      F: FnMut(Vec<usize>),
{
    trace!("Flusher running...");
    while let Some(msg) = receiver.next().await {
	use FlusherMsg::*;
	let mut maybe_gc = |this: Arc<CacheInner<N,K>>| {
	    match this.maybe_gc() {
		Ok(slots) if slots.len() > 0 => {
		    purger(slots.iter().map(|slot| (*slot).into()).collect())
		},
		Ok(_slots) => {
		    // This might indicate a problem
		    warn!("GC Unsuccessful?")
		},
		Err(e) => match e {
		    Error::LRUBackoff => trace!("LRU Backoff"),
		    _ => warn!("GC Error {:?}", e),
		}
	    }
	};
	match msg {
	    GC => {
		if let Some(this) = this.upgrade() {
		    trace!("Opportunistic GC");
		    maybe_gc(this);
		}
	    },
	    GcSync(sender) => {
		trace!("Sync GC requested");
		if let Some(this) = this.upgrade() {
		    maybe_gc(this)
		} else {
		    // Shouldn't happen, somewhere the waiter didn't wait.
		    warn!("Cache disappeared under flusher!");
		}
		// Really, just crash if the channel fails.. something is really bad
		sender.send(()).expect("Failed to wake up waiter");
	    },
	    SHUTDOWN => break,
	}
    }
    debug!("Stopping flusher for cache");

    // Generally the shutdown should only happen on drop,
    // so this should not be run anyway; but just to be future-proof..
    if let Some(this) = this.upgrade() {
	warn!("Shutting down flusher while cache '{}' still active..", this.name);
	let mut rw = this.rw.write().unwrap();
	// Give it back
	rw.flusher_receiver = Some(receiver);
    }
}


impl<const N: usize, const K: usize> Cache<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized {
    pub fn new(manager: &CacheManager<N,K>, name: &str, config: CacheConfig<N>) -> Result<Cache<N, K>> {
	let manager = manager.clone();
	debug!("New cache: {}, size: {} blocks", name, config.blocks);
	// Store everything in the DB, we get transactions "for free".
	let rmap_name = format!("{}!!rmap", name);
	let lru_name  = format!("{}!!lru", name);
	let tree = manager.db().open_tree(name)?;
	let rtree = manager.db().open_tree(rmap_name)?;
	let lru_tree = manager.db().open_tree(lru_name)?;
	let name = String::from(name);
	let lru = {
	    if let Ok(Some(iv_meta)) = lru_tree.get(LRU_META_KEY) {
		trace!("Loading LRU from disk");
		// Opening existing LRU
		LruArray::<N>::from_loader(iv_meta.as_ref(), |num, buf: &mut [u8; LRU_PAGE_SIZE]| {
		    // TODO: Recovery from corrupt database, iv_meta should be there only
		    //       after a successful transaction
		    let iv = lru_tree.get(LruBlockKey::from(num))?
			.ok_or_else(
			    || Error::CorruptError(format!("Block {} is missing from LRU. Database corrupt!", num)))?;
		    let src: &[u8] = iv.as_ref();
		    if src.len() != buf.len() {
			return Err(Error::CorruptError(format!("Block {} is truncated to {} bytes. Database corrupt!",
							       num, src.len())));
		    }
		    buf.as_mut_slice().copy_from_slice(iv.as_ref());
		    // Mixing some error types, make sure it's "our" own error collection type.
		    Ok::<(), Error>(())
		})?
	    } else {
		// New LRU: create LRU and save to disk
		// Run in transaction to make sure the LRU data stays consistent
		trace!("Creating new LRU");
		lru_tree.transaction(|tx| {
		    let mut lru = LruArray::new(&config);
		    // Save initial lru setup in DB
		    let (lru_meta, lru_iter) = lru.save();
		    for (num, page) in lru_iter.enumerate() {
			tx.insert(LruBlockKey::from(num).as_ref(), page.as_slice())?;
		    }
		    tx.insert(LRU_META_KEY.as_ref(), lru_meta)?;
		    Ok(lru)
		}).map_err(|e| match e {
		    // The abort could be some other type; we are however not using abort().
		    sled::transaction::TransactionError::Abort(db_err) => Error::DbError(db_err),
		    sled::transaction::TransactionError::Storage(db_err) => Error::DbError(db_err),
		})?
	    }
	};
	// The free hystesis point is about when there will be arriving new free ops
	let hysteresis_blocks: usize =
	    ((100.0 - config.free_hysteresis_pct as f64) / 100.0 * config.free_slack as f64).floor() as usize;
	let (flusher_sender, flusher_receiver) = mpsc::channel::<FlusherMsg>(hysteresis_blocks);
	let flusher_receiver = Some(flusher_receiver);
	Ok(Cache(Arc::new(CacheInner {
	    name,
	    manager,
	    tree,
	    rtree,
	    lru_tree,
	    rw: RwLock::new(CacheConcurrent {
		lru,
		flusher_sender,
		flusher_receiver,
	    }),
	})))
    }

    /// Returns flusher function closure that must be run
    /// The user should provide a purger function to handle automatic
    /// cache evictions. The purger is called with a list of slots that have
    /// been free'd.
    /// Returns a future that sheds the references to the cache.
    pub fn run<F>(&self, purger: F) -> impl Future<Output = ()>
    where F: FnMut(Vec<usize>) {
	let receiver =  {
	    let mut rw = self.0.rw.write().unwrap();
	    debug!("Starting flusher for cache '{}'", self.0.name);
	    // Crash if run multiple times
	    rw.flusher_receiver.take().unwrap()
	};

	let weak_this = Arc::downgrade(&self.0);

	let func = async move {
	    flusher(weak_this, receiver, purger).await;
	};
	func
    }

    /// Insert key into LRU. Key collisions are not handled, so an insert will always
    /// return a new slot number wheras the old one will be re-used later.
    /// Returns slot number
    pub async fn insert(&self, key: &Key<K>, level: usize) -> Result<usize>
    {
	assert!(level < N);
	let inner = &self.0;
	loop {
	    let ret = inner.get_lru(key, level).and_then(|slot| Ok(slot.into()));
	    match ret {
		Ok(slot) => {
		    return Ok(slot);
		},
		Err(e) => {
		    match e {
			Error::LRUFull => {
			    trace!("LRU Full, waking GC");
			    inner.wake_gc_sync().await;
			}
			_ => {
			    trace!("Transaction failed ({:?}, aborting", e);
			    return Err(e);
			}
		    }
		},
	    }
	}
    }

    /// Get key from LRU
    /// @level Optional level to bump access to (else it will bump
    ///        to the current level it resides in).
    pub fn get(&self, key: &Key<K>, level: Option<usize>) -> Option<usize> {
	let inner = &self.0;
	let slot = inner.tree.get(key.as_ref()).ok()??;
	let slot = Slot::from(slot);
	// Bump slot
	let mut rw = inner.rw.write().unwrap();
	rw.lru.promote(slot, level);
	Some(usize::from(slot))
    }
}


struct CacheInner<const N: usize, const K: usize>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    name: String,
    manager: CacheManager<N,K>,
    /// Sled sub-keyspace for mapping key -> slot
    tree: sled::Tree,
    /// Sled sub-keyspace for mapping slot -> key
    rtree: sled::Tree,
    /// Sled sub-keyspace for lru block storage
    lru_tree: sled::Tree,
    rw: RwLock<CacheConcurrent<N>>,
}

impl<const N: usize, const K: usize> CacheInner<N, K>
    where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized {

    // Wake flusher for GC work and wait for completion
    async fn wake_gc_sync(&self) {
	let (sender, receiver) = oneshot::channel::<()>();
	let mut rw = self.rw.write().unwrap();
	let ret = rw.flusher_sender.try_send(FlusherMsg::GcSync(sender));
	drop(rw);
	// If sending fails, the queue is stuffed and we have a crash/lock in flusher
	if let Ok(_) = ret {
	    receiver.await.expect("Waiting for flusher failed?")
	}
    }

    /// Save LRU changes to DB
    fn sync(&self) -> Result<()> {
	let mut rw = self.rw.write().unwrap();
	let (lru_meta, lru_shadow) = rw.lru.checkpoint();
	let pages = lru_shadow.as_slice();
	if pages.len() > 0 {
	    trace!("Syncing {} pages", pages.len());
	    self.lru_tree.transaction(|tx_lru| {
		for (pg_num, data) in pages {
		    tx_lru.insert(LruBlockKey::from(*pg_num).as_ref(), data.as_slice())?;
		}
		tx_lru.insert(LRU_META_KEY.as_ref(), &*lru_meta)?;
		Ok(())
	    }).map_err(|e| e.into())
	} else {
	    Ok(())
	}
    }

    /// Transactional get LRU
    fn get_lru(&self, key: &Key<K>, level: usize) -> Result<Slot> {
	let mut rw = self.rw.write().unwrap();
	rw.lru.get(level)
	    .ok_or(Error::LRUFull)
	    .and_then(|slot| {
		let (lru_meta, lru_shadow) = rw.lru.checkpoint();
		(&self.tree, &self.rtree, &self.lru_tree)
		    .transaction(|(tx_tree, tx_rtree, tx_lru)| {
			let other_slot = tx_tree.insert(key.as_ref(), sled::IVec::from(slot))?;
			if let Some(iv) = other_slot {
			    // Key collision, key is inserted a second time
			    let slot: Slot = iv.into();
			    // TODO: Handle this in some way..
			    warn!("Key collision on slot {}", slot);
			    // LRU2 GC will take care of it in time..
			    // Remove here so that key mapping isn't removed when this slot runs out.
			    tx_rtree.remove(&slot.as_key())?;
			}
			tx_rtree.insert(&slot.as_key(), key.as_ref())?;
			for (pg_num, data) in lru_shadow.as_slice().iter() {
			    tx_lru.insert(LruBlockKey::from(*pg_num).as_ref(), data.as_slice())?;
			}
			tx_lru.insert(LRU_META_KEY.as_ref(), &*lru_meta)?;
			Ok(slot)
		    }).map_err(|e| e.into())
	    }).and_then(|slot| {
		if rw.lru.on_gc_hysteresis() {
		    trace!("LRU hysteresis, order GC");
		    // Try to wake flusher
		    let _ = rw.flusher_sender.try_send(FlusherMsg::GC);
		}
		Ok(slot)
	    })
    }

    /// Try to do GC
    /// Returns list of GC'd Slots, LRUBackoff if there is enough space in the LRU
    /// Or other error happened during the DB update.
    fn maybe_gc(&self) -> Result<Vec<Slot>> {
	let mut rw = self.rw.write().unwrap();
	rw.lru.maybe_gc()
	    .ok_or(Error::LRUBackoff)
	    .and_then(|list| {
		let (lru_meta, lru_shadow) = rw.lru.checkpoint();
		(&self.tree, &self.rtree, &self.lru_tree)
		    .transaction(|(tx_tree, tx_rtree, tx_lru)| {
			let mut errors = 0;
			let mut slot_err = None;
			let mut key_err  = None;
			let slots: Vec<Slot> =
			    list.iter().map(|slot|
					    tx_rtree.remove(&slot.as_key())
					    .map_or_else(|e| {
						// We save one DB error
						errors += 1;
						slot_err = Some(e.clone());
						(*slot, None)
					    }, |some| (*slot, some)))
			    .filter(|(_, o)| o.is_some())
			    .map(|(slot, some)| {
				// If this errs, we are really in a broken DB
				// In key collisions the key -> slot mapping
				let _ = tx_tree.remove(some.unwrap())
				    .or_else(|e| {
					key_err = Some(e.clone());
					Err(e)
				    });
				slot
			    })
			    .collect();
			if slot_err.is_some() || key_err.is_some() {
			    warn!("DB Errors {} errors in total, {} keys removed",
				  errors, slots.len());
			    // TODO: Do something useful, Abort?
			}

			// Save LRU data
			for (pg_num, data) in lru_shadow.as_slice().iter() {
			    tx_lru.insert(LruBlockKey::from(*pg_num).as_ref(), data.as_slice())?;
			}
			tx_lru.insert(LRU_META_KEY.as_ref(), &*lru_meta)?;

			// Only return the successful ones
			// Not sure if we could fall into a case where LRU and DB disagrees though..
			Ok(slots)
		    }).or_else(|e| Err(e.into()))
	    })
    }
}

impl<const N: usize, const K: usize> Drop for CacheInner<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    fn drop(&mut self) {
	debug!("Cache {} shutting down", self.name);
	self.manager.purge_cache_cb(&self.name);
	// Sync cache to disk
	if let Err(e) = self.sync() {
	    // Not much one can do..
	    error!("Unable to sync LRU to disk on closing: {:?}", e);
	}
	// Shut down flusher
	let mut rw = self.rw.write().unwrap();
	if let Err(_) = rw.flusher_sender.try_send(FlusherMsg::SHUTDOWN) {
	    error!("Cache ({}) couldn't shutdown flusher", self.name);
	}
    }
}

struct CacheConcurrent <const N: usize>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    lru: LruArray<N>,
    /// Wakeup signal to flusher task
    flusher_sender: mpsc::Sender<FlusherMsg>,
    /// Flusher task
    flusher_receiver: Option<mpsc::Receiver<FlusherMsg>>,
}


/// LRU Block key helper. Allows us to access the underlying bits.
/// LRU block keys are signed as we use -1 as the meta key.
struct LruBlockKey(i32);

impl From<i32> for LruBlockKey {
    fn from(v: i32) -> LruBlockKey {
	assert!(v >= -1);
	LruBlockKey(v)
    }
}

impl From<usize> for LruBlockKey {
    fn from(v: usize) -> LruBlockKey {
	LruBlockKey(i32::try_from(v).unwrap())
    }
}

impl AsRef<[u8]> for LruBlockKey {
    fn as_ref(&self) -> &[u8] {
	let ptr = (&self.0) as *const i32 as *const u8;
	// Unsafe: Creating fat ptr to [u8] slice over the memory of an i32 (4 bytes).
	unsafe {
	    std::slice::from_raw_parts(ptr, std::mem::size_of::<i32>())
	}
    }
}

const LRU_META_KEY: LruBlockKey = LruBlockKey(-1);
