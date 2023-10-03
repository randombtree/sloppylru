/// PagedVec:
/// Vec-like continous array supporting page-out and shadow pages needed for
/// transactional commits.

use std::alloc;
use std::alloc::Layout;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

use log::{
    trace,
};

pub struct PagedVec<T, const N: usize>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    /// Pointer to continous array of at least sizeof(T) * capacity
    ptr: NonNull<T>,
    /// Requested capacity
    capacity: usize,
}


impl <T, const N: usize> PagedVec<T,N>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    pub fn new<F>(capacity: usize, mut initializer: F) -> PagedVec<T,N>
    where
	F: FnMut(usize) -> T,
    {
	assert!(capacity > 0, "Zero sized PageVec not supported");
	let layout = Self::layout(capacity);
	// unsafe: Allocation checked with NonNull below
	let ptr = unsafe {
	    std::alloc::alloc(layout)
	};
	let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };
	// Initialize data
	let data_ptr = ptr.as_ptr();
	for i in 0..capacity {
	    let val = initializer(i);
	    // unsafe: iterating inside capacity bounds
	    unsafe { data_ptr.add(i).write(val); };
	}
	PagedVec {
	    ptr,
	    capacity,
	}
    }

    pub fn len(&self) -> usize { self.capacity }

    /// Take multiple mutable references to array.
    /// indexes: A SORTED! list of indexes, without duplicates.
    pub fn take_refs_mut<'a, const S: usize>(&'a mut self, indexes: &[usize; S]) -> [&'a mut T;S]
    where
	[&'a mut T; S]: Sized,
    {
	let mut last_ndx: Option<usize> = None;
	let out = std::array::from_fn(|i| {
	    let index = indexes[i];
	    if index >= self.capacity {
		panic!("PagedVec: Index out of bounds! {} >= {}", index, self.capacity);
	    }
	    match last_ndx {
		Some(last) if last >= index => panic!("PagedVec: invalid index sort order!"),
		_ => ()
	    }
	    last_ndx = Some(index);
	    // TODO: Shadow handling
	    // unsafe: Protected by index bounds check above
	    unsafe { &mut *self.ptr.as_ptr().add(index) }
	});
	out
    }


    fn layout(capacity: usize) -> Layout {
	// Allocate page-aligned; the usual alignment should probably be system page alignment (e.g. 4096)
	// or at minimum IO-block sized (e.g. 512 bytes - which is mostly deprecated in favour of system page size
	// anyway)
	assert!(N < isize::MAX as usize, "Page size larger than maximum allocation unit!?");
	// It could be "easy" to do it, but it's not needed ATM:
	assert!(N & mem::size_of::<T>() == 0, "Non-page aligned item layouts not supported");
	let size = capacity * mem::size_of::<T>();
	let aligned_size = size / N * N + N * { if size % N == 0 { 0 } else { 1 }};
	trace!("Layout for max {} bytes, {} bytes total", size, aligned_size);
	Layout::from_size_align(aligned_size, N).unwrap()
    }
}

// PagedVec can safely be sent to other threads, no thread local storage used.
unsafe impl<T: Send, const N: usize> Send for PagedVec<T, N>
    where
    T: Sized  + Clone,
[T; N]: Sized,
{}

/// PagedVec references can be shared between threads.
unsafe impl<T: Sync, const N: usize> Sync for PagedVec<T, N>
    where
    T: Sized  + Clone,
[T; N]: Sized,
{}

impl<T, const N: usize> Index<usize> for PagedVec<T, N>
    where
    T: Sized  + Clone,
[T; N]: Sized,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
	if index >= self.capacity {
	    panic!("PagedVec: Index out of bounds, {} >= {}", index, self.capacity);
	}
	// unsafe: Protected by above check
	unsafe {
	    & *self.ptr.as_ptr().add(index)
        }
    }
}

impl<T, const N: usize> IndexMut<usize> for PagedVec<T, N>
    where
    T: Sized  + Clone,
[T; N]: Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
	if index >= self.capacity {
	    panic!("PagedVec: Index out of bounds, {} >= {}", index, self.capacity);
	}
	// unsafe: Protected by above check
	unsafe {
	    // TODO: Shadow
	    &mut *self.ptr.as_ptr().add(index)
        }
    }
}


impl<T, const N: usize> Drop for PagedVec<T, N>
    where
    T: Sized  + Clone,
{
    fn drop(&mut self) {
	let layout = Self::layout(self.capacity);
	// TODO: Currently not calling drop on contents; should need arise, this
	//       is where the call should happen :)

	// unsafe: self.ptr is guaranteed to be allocated in new
	unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}


mod tests {
    use super::PagedVec;

    #[derive(Clone, Copy)]
    struct TestStruct {
	foo: usize,
    }

    const PAGE_SIZE: usize = 4096;
    fn create_pvec() -> PagedVec<TestStruct, PAGE_SIZE> {
	const CAPACITY: usize = 10 * PAGE_SIZE;
	PagedVec::<TestStruct, PAGE_SIZE>::new(CAPACITY, |foo| TestStruct {foo})
    }

    #[test]
    fn test_create() {
	let vec = create_pvec();
	for i in 0 .. vec.len() {
	    assert!(vec[i].foo == i);
	}
    }

    #[test]
    #[should_panic]
    fn test_take_refs_invalid() {
	let mut vec = create_pvec();
	let [_two, _one] = vec.take_refs_mut(&[2,1]);
    }

    #[test]
    fn test_take_refs() {
	let mut vec = create_pvec();
	let [one, two, three] = vec.take_refs_mut(&[1,2,3]);
	assert!(one.foo == 1);
	assert!(two.foo == 2);
	assert!(three.foo == 3);
    }
}
