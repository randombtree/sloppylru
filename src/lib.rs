#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
/// Sloppy LRU database
/// ```
/// #![feature(generic_const_exprs)]
/// #![feature(adt_const_params)]
/// # use std::path::Path;
/// # use sloppylru::{CacheManager, Result, Cache, CacheConfig};
/// # use sloppylru::test::TestPath;
/// #
/// # fn try_create<P: AsRef<Path>>(db_path: P) -> Result<CacheManager<2, 64>> {
/// let manager = CacheManager::<2,64>::open(db_path)?;
/// let cache = manager.new_cache("my_cache", CacheConfig::default(1000))?;
/// # Ok(manager)
/// # }
/// #
/// # fn main() {
/// #    let test_path = TestPath::new("cache_manager_test_open").unwrap();
/// #    let db_path   = test_path.as_ref().join("testdb");
/// #    try_create(&db_path).unwrap();
/// # }
/// ```


mod cache;
mod config;

mod lru;
mod pagedvec;
pub mod key;
pub mod manager;
pub mod result;

pub use manager::CacheManager;
pub use config::CacheConfig;
pub use cache::Cache;

pub use result::{Result, Error};
// DocTest needs this, so need to live with it
pub mod test;


#[cfg(test)]
mod tests {
    use std::sync::Once;
    use log::info;

    use rand;
    use smol;
    use speculate::speculate;

    use super::*;
    use crate::key::Key;
    use crate::test::TestPath;


    static INIT: Once = Once::new();

    fn initialize() {
        INIT.call_once(|| {
            // Simple stdout logger, use RUST_LOG to set debug output (e.g. RUST_LOG=sloppylru=trace)
	    // let _ = env_logger::builder().is_test(true).init();
	    // NB: Only visible when test fails
        });
    }

    fn gen_key<const K: usize>() -> Key<K> {
	//let mut v = Vec::with_capacity(K);
	let v: Vec<u8> = (0..K).map(|_| rand::random::<u8>()).collect();
	Key::try_from(v).unwrap()
    }

    /// Blocking insert wrapper
    fn insert_key<const N: usize, const K: usize>(cache: &Cache<N, K>, key: &Key<K>, level: usize) -> usize
	where [(); 14 - N]: Sized, [(); N + 3]: Sized, [(); N + 2]: Sized
    {
	let ret = smol::block_on(async {
	    let ret = cache.insert(key, level).await;
	    std::result::Result::<usize, ()>::Ok(ret)
	}).unwrap();
	ret
    }

    speculate! {
	before {
	    initialize();
	}

	it "can create new lru cache" {
	    info!("Logging");
	    let db_path = TestPath::new("test_create_lru").unwrap();
	    assert!(db_path.as_ref().is_dir());
	    let manager = CacheManager::<2, 64>::open(db_path.as_ref()).unwrap();
	    let cache = manager.new_cache("my_cache",
					  CacheConfig::default(1000)).unwrap();
	    let key = gen_key();

	    let slot = insert_key(&cache, &key, 1);
	    let slot2 = cache.get(&key, Some(1)).unwrap();
	    assert!(slot == slot2);
	}
    }

}
