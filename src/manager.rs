use std::borrow::Borrow;
use std::collections::HashMap;
use std::str;
use std::fs;
use std::sync::{Arc, RwLock};
use log::{
    debug,
    trace,
    log_enabled,
    Level::Debug,
};

use sled;

use crate::result::{Result, Error};
use crate::config::CacheConfig;
use crate::cache::{Cache, WeakCache};

/// Cache manager for a shared backend database
pub struct CacheManager<const N: usize, const K: usize>(Arc<CacheManagerInner<N, K>>)
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized;

impl<const N: usize, const K: usize> Clone for CacheManager<N,K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    fn clone(&self) -> Self {
	CacheManager(self.0.clone())
    }
}

pub struct CacheManagerInner<const N: usize, const K: usize>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized {
    db:   sled::Db,
    caches: RwLock<HashMap<String, WeakCache<N, K>>>,
}

impl<const N: usize, const K: usize> CacheManager<N, K>
where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
{
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<CacheManager<N, K>> {
	let display_path = path.as_ref().to_string_lossy();
	debug!("Opening cache at {}, levels = {}", display_path, N);
	let path = path.as_ref().to_path_buf();
	if !path.is_dir() {
	    trace!("Creating {:?}", path);
	    fs::create_dir(&path)?;
	}
	let db_name = path.join("keys.sled");

	let db = sled::open(&db_name)?;

	let caches = RwLock::new(HashMap::new());

	if log_enabled!(Debug) {
	    let names_vec = db.tree_names();
	    match names_vec.len() {
		0 =>  debug!("DB has no caches"),
		_ =>  {
		    debug!("Contains the following caches:");
		    for name in names_vec.iter() {
			// Names are always in UTF8
			let name: &str = str::from_utf8(name.borrow()).unwrap();
			debug!("{}", name);
		    }
		}
	    }
	}

	Ok(CacheManager(Arc::new(CacheManagerInner {
	    db,
	    caches,
	})))
    }

    pub fn open_cache(&self, name: &str) -> Result<Cache<N, K>> {
	let caches = self.0.caches.write().unwrap();
	// Treat open/remove race as a bug for the moment,
	// there really shouldn't be any use case for dropping the cache
	// and re-opening it simultaneously from another thread
	match caches.get(name) {
	    Some(weak) => Ok(weak.try_into().unwrap()),
	    _ => Err(Error::SlruError(String::from("Not yet implemented"))),
	}
    }

    pub fn new_cache(&self, name: &str, config: CacheConfig<N>) -> Result<Cache<N, K>> {
	let mut caches = self.0.caches.write().unwrap();
	match caches.contains_key(name) {
	    true => Err(Error::CacheExists),
	    false => {
		let cache = Cache::new(&self, name, config)?;
		caches.insert(name.to_string(), WeakCache::new(&cache));
		Ok(cache)
	    }
	}
    }

    /// Call-back from Cache when dropped
    pub(crate) fn purge_cache_cb(&self, name: &str) {
	let mut caches = self.0.caches.write().unwrap();
	debug!("Cache '{}' is being dropped", name);
	if let Some(cache) = caches.remove(name) {
	    // Do some extra sanity checking
	    assert!(<WeakCache<N, K> as TryInto<Cache<N,K>>>::try_into(cache).is_err(),
		    "Cache still has references?");
	}
    }

    pub(crate) fn db(&self) -> &sled::Db { &self.0.db }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::Mutex;

    use futures::future::Future;
    use futures::join;
    use rand;
    use smol;
    use test_log::test;

    use log::{
	trace,
    };

    use crate::cache::Cache;
    use crate::config::CacheConfig;
    use crate::key::Key;
    use crate::lru;
    use crate::result::Result;
    use crate::test::TestPath;
    use super::CacheManager;


    const LEVELS: usize    = 2;
    const KEY_BYTES: usize = 64;
    const CACHE_SIZE_PAGES: usize = 10;
    type MyManager = CacheManager<LEVELS, KEY_BYTES>;
    type MyCache   = Cache<LEVELS, KEY_BYTES>;

    fn open_manager<P: AsRef<Path>>(path: P) -> Result<MyManager> {
	MyManager::open(path)
    }

    fn gen_key<const K: usize>() -> Key<K> {
	let v: Vec<u8> = (0..K).map(|_| rand::random::<u8>()).collect();
	Key::try_from(v).unwrap()
    }

    fn get_config(pages: usize) -> CacheConfig<LEVELS> {
	CacheConfig::default(pages * lru::ITEMS_PER_PAGE)
    }

    async fn open_cache<F>(manager: &MyManager, name: &str, purger: F) -> (MyCache, impl Future<Output = ()>)
	where F: FnMut(Vec<usize>) + Send +'static,
    {
	let cache = manager.new_cache(name, get_config(CACHE_SIZE_PAGES)).unwrap();
	let flusher = cache.run(purger);
	let task = smol::spawn(async move { flusher.await });
	(cache, task)
    }

    #[test]
    fn test_manager() {
	smol::block_on(async {
	    let config  = get_config(CACHE_SIZE_PAGES);
	    let db_path = TestPath::new("test_manager").unwrap();
	    let manager = open_manager(db_path).unwrap();
	    macro_rules! open_cache {
		($purger:expr) => {
		    open_cache(&manager, "my_cache", $purger).await
		};
		() => {
		    open_cache!(|list| trace!("Freed {:?}", list))
		};
	    }
	    let (cache, flusher) = open_cache!();
	    let (cache2, flusher2) = open_cache(&manager, "other_cache", |_l| {}).await;
	    let key = gen_key();
	    let key2 = gen_key();
	    let slot = cache.insert(&key, 1).await.unwrap();
	    let slot2 = cache2.insert(&key2, 1).await.unwrap();
	    // The insertion method is deterministic at the moment, so both slots should be the "same".
	    assert!(slot == slot2);
	    macro_rules! verify_key {
		($cache: ident, $key:ident, $slot:expr $(, $opt:expr)*) => {
		    let slot2 = $cache.get(&$key, None);
		    assert!(slot2.is_some(), $($opt),*);
		    assert!($slot == slot2.unwrap(), $($opt),*);
		}
	    }
	    // Inserted, should be found
	    verify_key!(cache, key, slot, "Key not found in cache after insert?");
	    verify_key!(cache2, key2, slot2);
	    // But neither should be found in it's sibling cache
	    assert!(cache.get(&key2, None).is_none());
	    assert!(cache2.get(&key, None).is_none());

	    // Test re-loading
	    drop(cache);
	    flusher.await;
	    let inserted_items = Arc::new(Mutex::new(HashMap::new()));
	    let removed_items  = Arc::new(Mutex::new(HashMap::new()));
	    let inserted_dup = inserted_items.clone();
	    let removed_dup  = removed_items.clone();

	    // Also test purger method
	    let (cache, flusher) = open_cache!(move |list| {
		let mut inserted_items = inserted_dup.lock().unwrap();
		let mut removed_items  = removed_dup.lock().unwrap();
		for slot in list {
		    let key = inserted_items.remove(&slot).expect("Removed item wasn't inserted?");
		    removed_items.insert(slot, key);
		}

	    });
	    verify_key!(cache, key, slot, "Key not found after reload?");
	    inserted_items.lock().unwrap().insert(slot, key);
	    // Fill'er up
	    // This should fill the whole LRU kicking "key" inserted above away
	    // TODO: waiting for a sequential async map ..
	    trace!("TEST GC: Filling up");
	    // The calculation here is to at least provoke a GC, but not too much to
	    // kick the newly inserted items out.
	    let insert_count = lru::ITEMS_PER_PAGE * CACHE_SIZE_PAGES - config.free_slack/2 + 1;
	    for _ in 0..insert_count {
		let key = gen_key();
		let slot = cache.insert(&key, 1).await.unwrap();
		let mut items = inserted_items.lock().unwrap();
		let mut removed = removed_items.lock().unwrap();
		items.insert(slot, key);
		// The last item will at least be there
		removed.remove(&slot);
	    }

	    assert!(removed_items.lock().unwrap().len() > 0);

	    trace!("TEST GC: Validating keys");
	    for (slot, key) in inserted_items.lock().unwrap().iter() {
		let ret = cache.get(key, None).expect("Expected key to be in LRU");
		assert!(*slot == ret, "Cache slot number mismatch!?");
	    }

	    drop(cache);
	    drop(cache2);
	    join!(flusher, flusher2);
	});
    }

    #[test]
    fn test_manager_reload() {
	smol::block_on(async {
	    let db_path = TestPath::new("test_manager_reload").unwrap();
	    let manager = open_manager(&db_path).unwrap();
	    let (cache, flusher) = open_cache(&manager, "my_cache", |_| {}).await;
	    let key = gen_key();
	    let slot = cache.insert(&key, 1).await.unwrap();
	    drop(cache);
	    flusher.await;
	    drop(manager);

	    // Re-open manager
	    let manager = open_manager(&db_path).unwrap();
	    let (cache, flusher) = open_cache(&manager, "my_cache", |_| {}).await;
	    let same_slot = cache.get(&key, None).unwrap();
	    assert!(slot == same_slot);

	    drop(cache);
	    flusher.await;
	});
    }
}
