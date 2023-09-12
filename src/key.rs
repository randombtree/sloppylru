/// slru key abstractions
use std::str;

use sled::IVec;


pub struct Key<const K: usize>(IVec) where
    [(); K]: Sized;

// Remove if unused
impl<const K: usize> Into<IVec> for Key<K> {
    fn into(self) -> IVec { self.0 }
}

/// sled uses ref for keys
impl<const K: usize> AsRef<IVec> for Key<K> {
    fn as_ref(&self) -> &IVec { &self.0 }
}

impl <const K: usize> TryFrom<IVec> for Key<K> {
    type Error = &'static str;
    fn try_from(ivec: IVec) -> core::result::Result<Self, Self::Error> {
	if ivec.len() == K {
	    Ok(Key(ivec))
	} else {
	    Err(concat!("Expected key length ", stringify!(N)))
	}
    }
}

impl <const K: usize> TryFrom<Vec<u8>> for Key<K> {
    type Error = &'static str;
    fn try_from(v: Vec<u8>) -> core::result::Result<Self, Self::Error> {
	if v.len() == K {
	    Ok(Key(IVec::from(v)))
	} else {
	    Err(concat!("Expected key length ", stringify!(N)))
	}
    }
}
