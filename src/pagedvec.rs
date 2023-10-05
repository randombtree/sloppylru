/// PagedVec:
/// Vec-like continous array supporting page-out and shadow pages needed for
/// transactional commits.

use std::alloc;
use std::alloc::Layout;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use log::{
    trace,
};

use bit_vec::BitVec;


/// Track shadow pages and current pages.
/// Also lifetime tracing for main memory (see Drop impl).
struct ShadowRegisterInner<T, const N: usize>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    /// The shadowed pages are actively being used (e.g. written to nv-storage).
    shadow_active: bool,
    /// Indicates that there are "current" pages in used
    current_present: bool,
    /// Pages taken out for writeback.
    shadowed_pages: BitVec,
    /// The "current" pages being written into.
    current_map: HashMap<usize, NonNull<T>>,
    /// The backing store in PagedVec, lifetime controlled in ArcMemory
    ptr: NonNull<T>,
    /// Capacity from PagedVec
    capacity: usize,
}

impl <T, const N: usize> ShadowRegisterInner<T,N>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    /// Current page allocation layout
    const LAYOUT: Layout = match Layout::from_size_align(N, N) {
	Ok(l) => l,
	Err(_) => panic!("Bad page size"),
    };

    /// Layout for backing array
    fn memory_layout(capacity: usize) -> Layout {
	// Allocate page-aligned; the usual alignment should probably
	// be system page alignment (e.g. 4096) or at minimum IO-block
	// sized (e.g. 512 bytes - which is mostly deprecated in
	// favour of system page size anyway)
	assert!(N < isize::MAX as usize, "Page size larger than maximum allocation unit!?");
	// It could be "easy" to do it, but it's not needed ATM:
	assert!(N & mem::size_of::<T>() == 0, "Non-page aligned item layouts not supported");
	let size = capacity * mem::size_of::<T>();
	let aligned_size = size / N * N + N * { if size % N == 0 { 0 } else { 1 }};
	trace!("Layout for max {} bytes, {} bytes total", size, aligned_size);
	Layout::from_size_align(aligned_size, N).unwrap()
    }


    /// Get pointer to current page if page is shadowed.
    fn get_page_ptr(&mut self, pgnum: usize) -> Option<*mut T> {
	if (self.shadow_active || self.current_present) && self.shadowed_pages[pgnum] {
	    let page = {
		// Page is being shadowed, is there already a current page mapped?
		if let Some(page) = self.current_map.get(&pgnum) {
		    *page
		} else {
		    // Unsafe: create a new page (free'd in !shadow_active below)
		    let ptr = unsafe { std::alloc::alloc(Self::LAYOUT) };
		    let ptr = match NonNull::new(ptr as *mut T) {
			Some(p) => p,
			None => alloc::handle_alloc_error(Self::LAYOUT),
		    };
		    // Copy shadow data to new page, use byte addressing for simplicity
		    // Unsafe: src is guaranteed to be there, copying a page is always safe due to
		    //         allocation granularity of one page.
		    let u8dst = ptr.as_ptr() as *mut u8;
		    let u8src = unsafe { (self.ptr.as_ptr() as *const u8).add(pgnum * N)};
		    // PageVec ends in full pages even though capacity is less, safe to copy whole page
		    unsafe { ptr::copy(u8src, u8dst, N); };
		    self.current_map.insert(pgnum, ptr);
		    self.current_present = true;
		    ptr
		}
	    };
	    Some(page.as_ptr())
	} else {
	    None
	}
    }

    /// Consolidate shadow pages with current pages if possible.
    /// This will make ALL REFS invalid, so make sure to only consolidate when
    /// no refs can be present to the PageVec array (i.e. taking a &mut self).
    unsafe fn maybe_consolidate(&mut self) {
	if self.current_present && !self.shadow_active {
	    // Shadow isn't active anymore but there are current-pages that need to be
	    // written back and discarded.
	    let backing_ptr = self.ptr.as_ptr() as *mut u8;
	    for (pgnum, page) in self.current_map.drain() {
		// Copy current page to permanent page, using byte-addressing for simplicity.
		// Unsafe: When keeping to page sizes we are safe; backing can't disappear.
		let u8src = page.as_ptr() as *const u8;
		let u8dst = unsafe { backing_ptr.add(pgnum * N) };
		unsafe { ptr::copy(u8src, u8dst, N); }
		// Can't have any mutable references to this page anymore
		unsafe {
		    alloc::dealloc(page.as_ptr() as *mut u8, Self::LAYOUT);
		}
	    }
	    // And clear all state
	    self.current_present = false;
	    self.shadowed_pages.clear();
	}
    }
}

impl<T, const N: usize> Drop for ShadowRegisterInner<T, N>
    where
    T: Sized  + Clone,
{
    fn drop(&mut self) {
	let layout = Self::memory_layout(self.capacity);
	// TODO: Currently not calling drop on contents; should need arise, this
	//       is where the call should happen :)
	trace!("Freeing PageVec memory");
	// unsafe: self.ptr is guaranteed to be allocated in new
	unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

/// An Arc to ShadowRegister will be both in ShadowPage and PagedVec
#[derive(Clone)]
struct ShadowRegister<T, const N: usize>(Arc<Mutex<ShadowRegisterInner<T, N>>>)
where
    T: Sized  + Clone,
[T; N]: Sized;

impl <T, const N: usize> ShadowRegister<T,N>
where
    T: Sized  + Clone,
[T; N]: Sized
{

    const ITEM_COUNT: usize = N / mem::size_of::<T>();

    fn new(capacity: usize, ptr: NonNull<T>) -> ShadowRegister<T, N> {
	let mut shadowed_pages = BitVec::with_capacity(capacity);
	shadowed_pages.grow(capacity, false);
	ShadowRegister(Arc::new(Mutex::new(ShadowRegisterInner {
	    shadow_active: false,
	    current_present: false,
	    shadowed_pages,
	    current_map: HashMap::new(),
	    ptr,
	    capacity,
	})))
    }

    /// Retrieve the current index pointers for the sorted list `indexes`. This might be to the backing
    /// store or a separate page if the page is being shadowed.
    /// If `consolidate` is set, will sync backing store if shadow isn't active anymore. Consolidation
    /// should only be done when no data refs are possible, e.g. in `take_refs_mut`.
    /// Generally this should only be called from PageVec index and take_refs_mut!!
    unsafe fn get_current_ptrs_unsafe<const S: usize>(&self, indexes: [usize; S], consolidate: bool) -> [*mut T;S] {
	let mut inner = self.0.lock().unwrap();
	if consolidate {
	    inner.maybe_consolidate();
	}

	let mut last_ndx: Option<usize> = None;
	let mut last_page: Option<(usize, *mut T)> = None;
	indexes.map(|index| {
	    if index >= inner.capacity {
		panic!("PagedVec: Index out of bounds! {} >= {}", index, inner.capacity);
	    }
	    match last_ndx {
		Some(last) if last >= index => panic!("PagedVec: invalid index sort order!"),
		_ => ()
	    }
	    let pgnum = index / Self::ITEM_COUNT;
	    // Cache last page to save a few cycles (in BitVec)
	    let page = match last_page {
		Some((last_num, ptr)) if last_num == pgnum => ptr,
		_ => {
		    if let Some(ptr) = inner.get_page_ptr(pgnum) {
			// Was shadowed
			ptr
		    } else {
			// Direct
			// Unsafe: pgnum is derived from index, checked earlier
			unsafe { inner.ptr.as_ptr().add(pgnum * Self::ITEM_COUNT)}
		    }
		}
	    };
	    last_page = Some((pgnum, page));
	    last_ndx = Some(index);
	    // Apply page offset
	    // Unsafe: Won't overflow page as ITEM_COUNT - 1 is page max index
	    unsafe { page.add(index % Self::ITEM_COUNT) }
	})
    }

    /// See comment on get_current_ptrs, this just masks away the "unsafe" consolidate.
    #[inline]
    fn get_current_ptrs<const S: usize>(&self, indexes: [usize; S]) -> [*mut T;S] {
	unsafe { self.get_current_ptrs_unsafe(indexes, false) }
    }

    /// Mark and return pages containing `indexes`, in order.
    /// indexes must be in sort order or we panic!
    fn grab_shadow_pages(&mut self, indexes: &[usize]) -> Vec<(usize, *const [T;N])> {
	let mut inner = self.0.lock().unwrap();
	if inner.shadow_active {
	    panic!("Only one shadow can be active at a time!");
	}
	// Make sure no pages lay around
	// Unsafe: We have &mut self, so no references can be active at the time
	unsafe { inner.maybe_consolidate(); }
	let mut last_ndx: Option<usize> = None;
	let mut last_page: Option<(usize, *const [T;N])> = None;
	let vec: Vec<(usize, *const [T;N])> = indexes.iter().map(|index| {
	    if *index >= inner.capacity {
		panic!("PagedVec: Index out of bounds! {} >= {}", index, inner.capacity);
	    }
	    match last_ndx {
		Some(last) if last >= *index => panic!("PagedVec: invalid index sort order!"),
		_ => ()
	    }
	    let pgnum = index / Self::ITEM_COUNT;
	    // We don't return duplicates
	    // Unsafe: pgnum checked from index
	    let ret = match last_page {
		Some((last_num, _ptr)) if last_num == pgnum => None,
		_ => Some((pgnum, unsafe { inner.ptr.as_ptr().add(pgnum * Self::ITEM_COUNT) as *const [T;N] }))
	    };
	    if ret.is_some() {
		last_page = ret;
		// Any access will now create a temporary "current" page.
		inner.shadowed_pages.set(pgnum, true);
	    }
	    last_ndx = Some(*index);
	    ret
	})
	    .filter_map(std::convert::identity)
	    .collect();
	if vec.len() > 0 {
	    inner.shadow_active = true;
	}
	vec
    }

    fn ungrab_shadow_pages(&self) {
	let mut inner = self.0.lock().unwrap();
	if !inner.shadow_active {
	    panic!("Shadow wasn't active, internal bug!");
	}
	inner.shadow_active = false;
	// Thats all folks! (consolidation happens in safer context).
    }
}

pub struct PagedVec<T, const N: usize>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    /// Memory reference count
    /// Requested capacity
    capacity: usize,
    /// "Overlay current" pages for shadowed pages.
    shadow: ShadowRegister<T, N>,
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
	let layout = ShadowRegisterInner::<T,N>::memory_layout(capacity);
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
	    capacity,
	    shadow: ShadowRegister::new(capacity, ptr),
	}
    }

    pub fn len(&self) -> usize { self.capacity }

    /// Get reference to item
    fn get(&self, ndx: usize) -> Option<&T> {
	if ndx >= self.capacity {
	    return None;
	}

	let mut index = ndx;
	let [ptr] = self.shadow.get_current_ptrs(*std::array::from_mut(&mut index));
	// Unsafe: get_current_ptrs always returns correct ptrs or panics.
	Some(unsafe { & *ptr })
    }

    /// Get mutable reference to item
    pub fn get_mut(&mut self, ndx: usize) -> Option<&mut T> {
	if ndx >= self.capacity {
	    return None;
	}

	let mut index = ndx;
	// Unsafe: Grabbing mutable reference, we can consolidate
	let [ptr] = unsafe {
	    self.shadow.get_current_ptrs_unsafe(*std::array::from_mut(&mut index), true)
	};
	// Unsafe: get_current_ptrs always returns correct ptrs or panics.
	Some(unsafe { &mut *ptr })
    }

    /// Take multiple mutable references to array.
    /// indexes: A SORTED! list of indexes, without duplicates.
    pub fn take_refs_mut<'a, const S: usize>(&'a mut self, indexes: [usize; S]) -> [&'a mut T;S]
    where
	[&'a mut T; S]: Sized,
    {
	// Unsafe: index is checked in get_current_ptrs, we have a &mut ref so other references
	//         can't be active - consolidation is possible
	unsafe {
	    self.shadow.get_current_ptrs_unsafe(indexes, true)
		.map(|ptr| &mut *ptr)
	}
    }

    /// Like take_refs_mut but does the extra work to allow unsorted indexes
    pub fn take_refs_mut_unsorted<'a, const S: usize>(&'a mut self, indexes: [usize; S]) -> [&'a mut T;S]
    {
	// Store original order
	let mut remap = {
	    let mut i = 0;
	    indexes.map(|index| {
		let ret = (i, index); i+= 1; ret
	    })
	};
	remap.sort_by_key(|item| item.1); // Sort by array index
	let sorted_indexes: [usize; S] = std::array::from_fn(|ndx| remap[ndx].1);

	// Now need to map order back to what was given
	let mut refs = {
	    let mut i = 0;
	    self.take_refs_mut(sorted_indexes).map(|mutref| {
		// Remap has the original order stored in the tuple
		let ret = (remap[i].0, mutref);
		i += 1;
		ret
	    })
	};
	refs.sort_by_key(|item| item.0);

	refs.map(|(_ndx, mutref)| mutref)
    }

    /// Takes shadow pages for the indexes specified.
    /// The indexes must be sorted!
    /// It's assumed that the caller has tracked changes to the indexes and now wants
    /// to write out the pages containing the indexes to stable storage.
    /// Returns `ShadowPages` that contains the references to the pages. Only one set
    ///         of ShadowPages can be present at the moment and will panic! if another set is active!
    pub fn take_shadow_pages(&mut self, indexes: &[usize]) -> ShadowPages<T, N> {
	let pages = self.shadow.grab_shadow_pages(indexes);
	ShadowPages::new(self.shadow.clone(), pages)
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
	self.get(index).expect("PagedVec: Index out of bounds")
    }
}

impl<T, const N: usize> IndexMut<usize> for PagedVec<T, N>
    where
    T: Sized  + Clone,
[T; N]: Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
	self.get_mut(index).expect("PagedVec: Index out of bounds")
    }
}


pub struct ShadowPages<T, const N: usize>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    /// Lock the shadow memory in place while we exist
    shadow: ShadowRegister<T,N>,
    /// The shadow pages, with pointer to the above memory
    pages: Vec<(usize, *const [T;N])>,
}

impl <T, const N: usize> ShadowPages<T,N>
where
    T: Sized  + Clone,
[T; N]: Sized,
{
    fn new(shadow: ShadowRegister<T,N>, pages: Vec<(usize, *const [T;N])>) -> ShadowPages<T,N> {
	ShadowPages {
	    shadow,
	    pages,
	}
    }

    pub fn as_slice<'a>(&'a self) -> &'a [(usize, &'a[T;N])] {
	// Unsafe: Convert pointer to reference
	unsafe { std::mem::transmute(self.pages.as_slice()) }
    }
}

impl<T, const N: usize> Drop for ShadowPages<T, N>
    where
    T: Sized  + Clone,
{
    fn drop(&mut self) {
	self.shadow.ungrab_shadow_pages();
    }
}


mod tests {
    use super::PagedVec;

    #[derive(Clone, Copy)]
    struct TestStruct {
	foo: usize,
    }


    const PAGE_SIZE: usize = 4096;
    const ITEMS_PER_PAGE: usize = PAGE_SIZE / std::mem::size_of::<TestStruct>();

    type PVec = PagedVec<TestStruct, PAGE_SIZE>;
    fn create_pvec() -> PVec {
	const CAPACITY: usize = 10 * PAGE_SIZE;
	PagedVec::<TestStruct, PAGE_SIZE>::new(CAPACITY, |foo| TestStruct {foo})
    }

    fn assert_contents(vec: &PVec) {
	for i in 0 .. vec.len() {
	    assert!(vec[i].foo == i);
	}

    }

    #[test]
    fn test_create() {
	let vec = create_pvec();
	assert_contents(&vec);
    }

    #[test]
    #[should_panic]
    fn test_take_refs_invalid_order() {
	let mut vec = create_pvec();
	let [_two, _one] = vec.take_refs_mut([2,1]);
    }

    #[test]
    #[should_panic]
    fn test_take_refs_duplicates() {
	let mut vec = create_pvec();
	let [_one1, _one2] = vec.take_refs_mut([1,1]);
    }


    #[test]
    fn test_take_refs() {
	let mut vec = create_pvec();
	let [one, two, three] = vec.take_refs_mut([1,2,3]);
	assert!(one.foo == 1);
	assert!(two.foo == 2);
	assert!(three.foo == 3);

	let [two, one, three] = vec.take_refs_mut_unsorted([2,1,3]);
	assert!(one.foo == 1);
	assert!(two.foo == 2);
	assert!(three.foo == 3);
    }

    #[test]
    fn test_shadow() {
	let mut vec = create_pvec();
	let indexes = [1, 2, 3,
		       2 * ITEMS_PER_PAGE + 5,
		       6 * ITEMS_PER_PAGE + 10,
	];
	let mut pgnums: Vec<usize> = indexes.iter().map(|ndx| ndx / ITEMS_PER_PAGE).collect();
	pgnums.dedup();
	let shadows = vec.take_shadow_pages(&indexes);
	let pages = shadows.as_slice();
	// The length is right
	assert!(pgnums.len() == pages.len());
	// Contains the right page indexes
	let pgnums2: Vec<usize> = pages.iter().map(|(ndx, _p)| *ndx).collect();
	assert!(pgnums == pgnums2.as_slice());

	macro_rules! match_shadow {
	    ($pages:ident) => {
		// The page references themselves are correct, page contents match
		for (ndx, pg) in $pages {
		    let start_index = ndx * ITEMS_PER_PAGE;
		    let end_index = start_index + ITEMS_PER_PAGE - 1;
		    for (pgindex, index) in (start_index ..=end_index).enumerate() {
			assert!(index == pg[pgindex].foo);
		    }
		}
	    }
	}

	match_shadow!(pages);
	// Old indexes should still match
	assert_contents(&vec);

	// Change the "current" page
	vec[1] = TestStruct { foo: 1000 };
	assert!(vec[1].foo == 1000);

	// It shouldn't affect the shadow:
	match_shadow!(pages);

	// Revert change above, but now we have a "current" page in the mix
	vec[1] = TestStruct { foo: 1 };
	// the old contents should match again
	assert_contents(&vec);

	// And now before the drop, make sure the change is propagated back
	vec[1] = TestStruct { foo: 1000 };
	match_shadow!(pages); // To keep shadow alive
	assert!(vec[1].foo == 1000);
	drop(shadows);
	// Trigger consolidation
	vec[2] = TestStruct { foo: 2000 };
	// And the consolidated result should still be there
	assert!(vec[0].foo == 0);
	assert!(vec[1].foo == 1000);
	assert!(vec[2].foo == 2000);
    }
}
