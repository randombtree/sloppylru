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
pub use lru::ITEMS_PER_PAGE;

pub use result::{Result, Error};
// DocTest needs this, so need to live with it
pub mod test;
