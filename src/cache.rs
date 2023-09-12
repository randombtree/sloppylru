/// SloppyLRU cache
/// @todo Use plain file + log for rmap transaction for faster insert speed

use std::fs;
use std::str;
use std::sync::{Arc, RwLock, Weak};
use futures::channel::{mpsc, oneshot};
use futures::future::Future;
use futures::stream::StreamExt;
use log::{
    debug,
    error,
    trace,
};

use sled;
use sled::Transactional;
use sled::transaction::{
    abort,
    ConflictableTransactionError
};

use crate::result::Result;
use crate::config::CacheConfig;
use crate::manager::CacheManager;
use crate::lru::{LruArray, Slot};
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
	// TODO: using a db for a continious space is really unneccesary and inefficient, but
	//       this way we get transactions for free for the moment
	let path = manager.path();
	assert!(path.is_dir());
	let path = path.join(name);
	if !path.is_dir() {
	    trace!("Create directory for cache: {:?}", path);
	    fs::create_dir(&path)?;
	}
	let rmap_name = format!("{}!!rmap", name);
	let tree = manager.db().open_tree(name)?;
	let rtree = manager.db().open_tree(rmap_name)?;
	let name = String::from(name);
	let lru = LruArray::new(&config);
	// The free hystesis point is about when there will be arriving new free ops
	let hysteresis_blocks: usize =
	    ((100.0 - config.free_hysteresis_pct as f64) / 100.0 * config.free_slack as f64).floor() as usize;
	let (flusher_sender, flusher_receiver) = mpsc::channel::<FlusherMsg>(hysteresis_blocks);
	let flusher_receiver = Some(flusher_receiver);
	Ok(Cache(Arc::new(CacheInner {
	    name,
	    config,
	    manager,
	    tree,
	    rtree,
	    flusher_sender,
	    rw: RwLock::new(CacheConcurrent {
		lru,
		flusher_receiver,
	    }),
	})))
    }

    /// Returns flusher function closure that must be run
    pub async fn run(&self) -> impl Future<Output = ()> {
	let this = self.0.clone();
	let mut receiver = Option::None;
	{
	    let mut rw = this.rw.write().unwrap();
	    debug!("Starting flusher for cache '{}'", this.name);
	    std::mem::swap(&mut rw.flusher_receiver, &mut receiver);
	}
	// Crash if run multiple times
	let mut receiver = receiver.unwrap();

	let func = async move {
	    while let Some(msg) = receiver.next().await {
		use FlusherMsg::*;
		match msg {
		    GC => { todo!() },
		    GC_SYNC(_sender) => { todo!() },
		    SHUTDOWN => break,
		}
	    }
	    // Give it back
	    let mut receiver = Option::Some(receiver);
	    {
		let mut rw = this.rw.write().unwrap();
		debug!("Stopping flusher for cache '{}'", this.name);
		std::mem::swap(&mut rw.flusher_receiver, &mut receiver);
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
		let res =
		    (&inner.tree, &inner.rtree)
		    .transaction(|(tx_tree, tx_rtree)| {
			tx_tree.insert(key.as_ref(), sled::IVec::from(slot))?;
			tx_rtree.insert(&slot.as_key(), key.as_ref())?;
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
    config: CacheConfig<N>,
    manager: CacheManager<N,K>,
    /// Sled sub-keyspace for mapping key -> slot
    tree: sled::Tree,
    /// Sled sub-keyspace for mapping slot -> key
    rtree: sled::Tree,
    /// Wakeup signal to flusher task
    flusher_sender: mpsc::Sender<FlusherMsg>,
    rw: RwLock<CacheConcurrent<N>>,
}

impl<const N: usize, const K: usize> Drop for CacheInner<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    fn drop(&mut self) {
	debug!("Cache {} shutting down", self.name);
	self.manager.purge_cache_cb(&self.name);
	// Shut down flusher
	if let Err(_) = self.flusher_sender.try_send(FlusherMsg::SHUTDOWN) {
	    error!("Cache ({}) couldn't shutdown flusher", self.name);
	}
    }
}

struct CacheConcurrent <const N: usize>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    lru: LruArray<N>,
    /// Flusher task
    flusher_receiver: Option<mpsc::Receiver<FlusherMsg>>,
}

