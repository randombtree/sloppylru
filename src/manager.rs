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
    path: std::path::PathBuf,
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
	    path,
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
    pub(crate) fn path(&self) -> &std::path::Path { &self.0.path.as_ref() }
}
