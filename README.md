# sloppylru - fast but lossy embedded multi-level LRU cache

## Design targets
For efficient caching in e.g. DHT:s that can cache far-away keys with lower priority. Can lose LRU promotions on unexpected shutdowns, but keeps the key-store consistent.
Still in early development, don't use!

## Usage

Create a two-level LRU cache:
```
let manager = CacheManager::<2,64>::open(db_path)?;
let cache = manager.new_cache("my_cache", CacheConfig::default(1000))?;
```

Insert and retrieve index of key item.
```
let index = cache.insert(key, level).await;
...
if let Some(index) = cache.get(key, Some(level)) {
    // Has key
}
```

