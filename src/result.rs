use std::io;
use sled;

/// Cache errors
#[derive(Debug)]
pub enum Error {
    /// Error from sled DB
    DbError(sled::Error),
    /// Error from IO subsystem
    IOError(io::Error),
     /// Data-corruption detected in the structures
    CorruptError(String),
    /// Two threads trying to insert the same key at the same time
    KeyCollision,
    /// SloppyLRU error
    SlruError(String),
    /// Trying to open non-existing cache
    CacheNotFound,
    /// Trying to create existing cache
    CacheExists,
}

impl From<sled::Error> for Error {
    fn from(error: sled::Error) -> Self {
	Error::DbError(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
	Error::IOError(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
