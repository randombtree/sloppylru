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
use sled::transaction::{
    abort,
};

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
    GC_SYNC(oneshot::Sender<()>),
    /// Shut down flusher
    SHUTDOWN,
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
    /// Returns a future that sheds the references to the cache.
    pub async fn run(&self) -> impl Future<Output = ()> {
	let mut receiver =  {
	    let mut rw = self.0.rw.write().unwrap();
	    debug!("Starting flusher for cache '{}'", self.0.name);
	    // Crash if run multiple times
	    rw.flusher_receiver.take().unwrap()
	};

	let weak_this = Arc::downgrade(&self.0);

	let func = async move {
	    trace!("Flusher running...");
	    while let Some(msg) = receiver.next().await {
		use FlusherMsg::*;
		match msg {
		    GC => { todo!() },
		    GC_SYNC(_sender) => { todo!() },
		    SHUTDOWN => break,
		}
	    }
	    debug!("Stopping flusher for cache");

	    // Generally the shutdown should only happen on drop,
	    // so this should not be run anyway; but just to be future-proof..
	    if let Some(this) = weak_this.upgrade() {
		warn!("Shutting down flusher while cache '{}' still active..", this.name);
		let mut rw = this.rw.write().unwrap();
		// Give it back
		rw.flusher_receiver = Some(receiver);
	    }
	};
	func
    }

    /// Insert key into LRU
    /// Returns slot number
    pub async fn insert(&self, key: &Key<K>, level: usize) -> usize
    {
	assert!(level < N);
	let inner = &self.0;
	// TODO: GC wakeup logic goes here
	loop {
	    // TODO: LRU slot transaction
	    let mut rw = inner.rw.write().unwrap();
	    if let Some(slot) = rw.lru.get(level) {
		// Get the lru changes up to now.
		let (lru_meta, lru_shadow) = rw.lru.checkpoint();
		let res =
		    (&inner.tree, &inner.rtree, &inner.lru_tree)
		    .transaction(|(tx_tree, tx_rtree, tx_lru)| {
			tx_tree.insert(key.as_ref(), sled::IVec::from(slot))?;
			tx_rtree.insert(&slot.as_key(), key.as_ref())?;
			for (pg_num, data) in lru_shadow.as_slice().iter() {
			    tx_lru.insert(LruBlockKey::from(*pg_num).as_ref(), data.as_slice())?;
			}
			tx_lru.insert(LRU_META_KEY.as_ref(), &*lru_meta)?;
			if false {
			    abort(())?;
			}
			Ok(usize::from(slot))
		    });
		if let Ok(slot) = res {
		    return slot;
		} else {
		    trace!("Transaction failed ({:?}, retrying...", res.unwrap_err());
		}
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

    /// Evict key from LRU
    pub fn evict(&self, key: &Key<K>) -> Option<()> {
	todo!();
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

impl<const N: usize, const K: usize> Drop for CacheInner<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    fn drop(&mut self) {
	debug!("Cache {} shutting down", self.name);
	self.manager.purge_cache_cb(&self.name);
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
