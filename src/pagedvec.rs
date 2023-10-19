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
use std::iter::Iterator;

use log::{
    trace,
};

use bit_vec::BitVec;


/// Track shadow pages and current pages.
/// Also lifetime tracing for main memory (see Drop impl).
struct ShadowRegisterInner<T, const N: usize>
where
    T: Sized  + Clone + 'static,
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
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    /// Current page allocation layout
    const LAYOUT: Layout = match Layout::from_size_align(N, N) {
	Ok(l) => l,
	Err(_) => panic!("Bad page size"),
    };

    const ITEMS_PER_PAGE: usize = N / mem::size_of::<T>();

    fn capacity_to_pages(capacity: usize) -> usize {
	capacity / Self::ITEMS_PER_PAGE + { if capacity % Self::ITEMS_PER_PAGE > 0 { 1 } else { 0 }}
    }

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
	let aligned_size = N * Self::capacity_to_pages(capacity);
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
    T: Sized  + Clone + 'static,
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
    T: Sized  + Clone + 'static,
[T; N]: Sized;

impl <T, const N: usize> ShadowRegister<T,N>
where
    T: Sized  + Clone + 'static,
[T; N]: Sized
{

    const ITEM_COUNT: usize = N / mem::size_of::<T>();

    fn new(capacity: usize, ptr: NonNull<T>) -> ShadowRegister<T, N> {
	let pages = capacity / Self::ITEM_COUNT
	    + { if capacity % Self::ITEM_COUNT > 0 { 1 }  else { 0 } };
	let mut shadowed_pages = BitVec::with_capacity(pages);
	shadowed_pages.grow(pages, false);
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
    fn grab_shadow_pages(&mut self, indexes: &[usize]) -> Vec<(usize, *const [u8;N])> {
	let mut inner = self.0.lock().unwrap();
	if inner.shadow_active {
	    panic!("Only one shadow can be active at a time!");
	}
	// Make sure no pages lay around
	// Unsafe: We have &mut self, so no references can be active at the time
	unsafe { inner.maybe_consolidate(); }
	let base_ptr = inner.ptr.as_ptr();
	let mut last_ndx: Option<usize> = None;
	let mut last_page: Option<(usize, *const [u8;N])> = None;
	let vec: Vec<(usize, *const [u8;N])> = indexes.iter().map(|index| {
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
		_ => Some((pgnum, unsafe {
		    // Unsafe: pgnum checked, get pointer to one page worth of bytes
		    base_ptr.add(pgnum * Self::ITEM_COUNT) as *const [u8;N]
		}))
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

	// Even if we return a 0-length array, i.e. indexes.len() == 0, we must activate the "shadow"
	// as it implies that there is a ShadowPages struct that will "free" the shadow pages later.
	inner.shadow_active = true;

	vec
    }
}

/// Decouple shadow content type from actual shadow
trait ShadowUngrabber {
    fn ungrab_shadow_pages(&self);
}

impl<T, const N: usize> ShadowUngrabber for ShadowRegister<T, N>
where
    T: Sized  + Clone + 'static,
[T; N]: Sized {
    fn ungrab_shadow_pages(&self) {
	let mut inner = self.0.lock().unwrap();
	if !inner.shadow_active {
	    panic!("Shadow wasn't active, internal bug!");
	}
	inner.shadow_active = false;
	// Thats all folks! (consolidation happens in safer context).
    }
}

/// Another shadow page decoupler, this time for iterating pages
trait PageGetter<const N: usize> {
    fn get_page(&self, index: usize) -> Option<*const u8>;
}

impl<T, const N: usize> PageGetter<N> for ShadowRegister<T, N>
where
    T: Sized  + Clone + 'static,
{
    fn get_page(&self, pg_index: usize) -> Option<*const u8> {
	let mut inner = self.0.lock().unwrap();
	let byte_offset = Self::ITEM_COUNT * pg_index;
	if byte_offset >= inner.capacity {
	    return None;
	}
	let ptr = match inner.get_page_ptr(pg_index) {
	    Some(ptr) => ptr,
	    // Unsafe: Byte offset checked
	    _ => unsafe { inner.ptr.as_ptr().add(pg_index * Self::ITEM_COUNT) }
	};
	// Unsafe: Guaranteed safe due to allocation granularity
	unsafe { Some(std::mem::transmute(ptr)) }
    }
}


pub struct PagedVec<T, const N: usize>
where
    T: Sized  + Clone + 'static,
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
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    fn alloc_mem(capacity: usize) -> NonNull<T> {
	assert!(capacity > 0, "Zero sized PageVec not supported");
	let layout = ShadowRegisterInner::<T,N>::memory_layout(capacity);
	// unsafe: Allocation checked with NonNull below
	let ptr = unsafe {
	    std::alloc::alloc(layout)
	};
	match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        }
    }

    fn dealloc_mem(capacity: usize, non_null: NonNull<T>) {
	let layout = ShadowRegisterInner::<T,N>::memory_layout(capacity);
	unsafe {
            alloc::dealloc(non_null.as_ptr() as *mut u8, layout);
        }
    }

    pub fn new<F>(capacity: usize, mut initializer: F) -> PagedVec<T,N>
    where
	F: FnMut(usize) -> T,
    {
	let ptr = Self::alloc_mem(capacity);
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

    /// Initialize PageVec from previously saved pages.
    pub fn from_loader<F, E>(capacity: usize, mut loader: F) -> Result<PagedVec<T,N>, E>
    where
	F: FnMut(usize, &mut [u8; N]) -> Result<(), E>
    {
	let non_null = Self::alloc_mem(capacity);
	let pg_count = ShadowRegisterInner::<T,N>::capacity_to_pages(capacity);
	let ptr = non_null.as_ptr();
	for i in 0..pg_count {
	    // Unsafe: handing out references to each page [u8; N] one page at a time.
	    //         Transmuting ptr to reference
	    let ret = loader(i, unsafe {
		std::mem::transmute(ptr.add(i * ShadowRegisterInner::<T, N>::ITEMS_PER_PAGE))
	    });
	    if let Err(e) = ret {
		trace!("Loader failed, dealloc memory");
		Self::dealloc_mem(capacity, non_null);
		return Err(e);
	    }
	}
	Ok(PagedVec {
	    capacity,
	    shadow: ShadowRegister::new(capacity, non_null),
	})
    }

    pub fn len(&self) -> usize { self.capacity }

    /// Get the number of pages in this PageVec
    pub fn pages(&self) -> usize { ShadowRegisterInner::<T,N>::capacity_to_pages(self.capacity) }

    /// Get reference to item
    pub fn get(&self, ndx: usize) -> Option<&T> {
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

    /// Get iterator over the values in this container
    pub fn iter<'a>(&'a self) -> Iter<'a, T, N>
    {
	Iter::new(&self)
    }

    /// Get a page iterator for write-output purposes.
    pub fn page_iter<'a>(&'a self) -> PageIter<'a, T, N> {
	let shadow = Box::new(self.shadow.clone());
	PageIter::new(shadow)
    }

    /// Takes shadow pages for the indexes specified.
    /// The indexes must be sorted!
    /// It's assumed that the caller has tracked changes to the indexes and now wants
    /// to write out the pages containing the indexes to stable storage.
    /// Returns `ShadowPages` that contains the references to the pages. Only one set
    ///         of ShadowPages can be present at the moment and will panic! if another set is active!
    pub fn take_shadow_pages(&mut self, indexes: &[usize]) -> ShadowPages<N> {
	let pages = self.shadow.grab_shadow_pages(indexes);
	ShadowPages::new(Box::new(self.shadow.clone()), pages)
    }
}

// PagedVec can safely be sent to other threads, no thread local storage used.
unsafe impl<T: Send, const N: usize> Send for PagedVec<T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{}

/// PagedVec references can be shared between threads.
unsafe impl<T: Sync, const N: usize> Sync for PagedVec<T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{}

impl<T, const N: usize> Index<usize> for PagedVec<T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
	self.get(index).expect("PagedVec: Index out of bounds")
    }
}

impl<T, const N: usize> IndexMut<usize> for PagedVec<T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
	self.get_mut(index).expect("PagedVec: Index out of bounds")
    }
}


/// PagedVec iterator over items in container
pub struct Iter<'a, T, const N: usize>
where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    index: usize,
    pagedvec: &'a PagedVec<T, N>,
}

impl<'a, T, const N: usize> Iter<'a, T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    fn new(pagedvec: &'a PagedVec<T, N>) -> Iter<'a, T, N> {
	Iter {
	    index: 0,
	    pagedvec,
	}
    }
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
	self.pagedvec.get(self.index).and_then(|i| {
	    self.index += 1;
	    Some(i)
	})
    }
}


pub struct PageIter<'a, T, const N: usize>
where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    getter: Box<dyn PageGetter<N>>,
    page_index: usize,
    phantom: std::marker::PhantomData<&'a PagedVec<T, N> >,
}

impl<'a, T, const N: usize> PageIter<'a, T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    fn new(getter: Box<dyn PageGetter<N>>) -> PageIter<'a, T, N> {
	PageIter {
	    getter,
	    page_index: 0,
	    phantom: std::marker::PhantomData,
	}
    }
}

impl<'a, T, const N: usize> Iterator for PageIter<'a, T, N>
    where
    T: Sized  + Clone + 'static,
[T; N]: Sized,
{
    type Item = &'a [u8; N];
    fn next(&mut self) -> Option<Self::Item> {
	if let Some(ptr) = self.getter.get_page(self.page_index) {
	    self.page_index += 1;
	    // Unsafe: Page is guaranteed to be there, locked by getter.
	    Some(unsafe { std::mem::transmute(ptr) })
	} else {
	    None
	}
    }
}


pub struct ShadowPages<const N: usize>
{
    /// Lock the shadow memory in place while we exist
    shadow: Box<dyn ShadowUngrabber>,
    /// The shadow pages, with pointer to the above memory
    pages: Vec<(usize, *const [u8;N])>,
}

impl <const N: usize> ShadowPages<N>
{
    fn new(shadow: Box<dyn ShadowUngrabber>, pages: Vec<(usize, *const [u8;N])>) -> ShadowPages<N> {
	ShadowPages {
	    shadow,
	    pages,
	}
    }

    pub fn as_slice<'a>(&'a self) -> &'a [(usize, &'a[u8;N])] {
	// Unsafe: Convert pointer to reference
	unsafe { std::mem::transmute(self.pages.as_slice()) }
    }
}

impl<const N: usize> Drop for ShadowPages<N>
{
    fn drop(&mut self) {
	self.shadow.ungrab_shadow_pages();
    }
}


mod tests {
    use log::trace;
    use super::PagedVec;

    #[derive(Clone, Copy)]
    struct TestStruct {
	foo: usize,
    }


    const PAGE_SIZE: usize = 4096;
    const ITEMS_PER_PAGE: usize = PAGE_SIZE / std::mem::size_of::<TestStruct>();
    const PAGES: usize = 10;
    const CAPACITY: usize = PAGES * ITEMS_PER_PAGE;

    type PVec = PagedVec<TestStruct, PAGE_SIZE>;
    fn create_pvec() -> PVec {
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
    fn test_index_write() {
	let mut vec = create_pvec();
	for i in 0..100 {
	    vec[i].foo = i + 100;
	}
	for i in 0..100 {
	    assert!(vec[i].foo == i + 100);
	}
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_read() {
	let vec = create_pvec();
	let _val = vec[CAPACITY].foo;
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_write() {
	let mut vec = create_pvec();
	vec[CAPACITY].foo = 1000;
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
		    assert!(pg.len() == PAGE_SIZE);
		    // Unsafe: We know there are ITEMS_PER_PAGE struct items..
		    let pg: &[TestStruct; ITEMS_PER_PAGE]
			= unsafe { std::mem::transmute(*pg) };

		    let start_index = ndx * ITEMS_PER_PAGE;
		    let end_index = start_index + ITEMS_PER_PAGE - 1;
		    for (pgindex, index) in (start_index ..=end_index).enumerate() {
			//println!("{} {} {} {}", ndx, pgindex, index, pg[pgindex].foo);
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

    #[test]
    fn test_iter() {
	let vec = create_pvec();
	let mut counter: usize = 0;
	for (index, item) in vec.iter().enumerate() {
	    assert!(index == item.foo);
	    counter += 1;
	}
	assert!(counter == CAPACITY, "Iterator did not return all items?");
    }

    #[test]
    fn test_page_iter() {
	let vec = create_pvec();
	let mut pgcount = 0;
	for (index, page) in vec.page_iter().enumerate() {
	    assert!(index < PAGES);
	    assert!(page.len() == PAGE_SIZE);
	    pgcount += 1;
	    let page: &[TestStruct; ITEMS_PER_PAGE] = unsafe { std::mem::transmute(page) };
	    for (offset, test) in page.iter().enumerate() {
		trace!("{} {} {}", index, offset, test.foo);
		assert!(test.foo == ITEMS_PER_PAGE * index + offset);
	    }
	}
	assert!(pgcount == PAGES);
    }

    #[test]
    fn test_from_loader() {
	let vec = create_pvec();
	let mut iter = vec.page_iter();
	let mut counter = 0;
	let loaded_pvec: PVec =  PVec::from_loader(CAPACITY, |ndx, dst| {
	    let src = iter.next().unwrap();
	    // Unsafe: Transmute to pointers to make copy happy and do the right thing..
	    let src: *const u8 = unsafe { std::mem::transmute(src) };
	    let dst: *mut u8 = unsafe { std::mem::transmute(dst) };
	    assert!(ndx == counter);
	    counter += 1;
	    unsafe {
		std::ptr::copy_nonoverlapping(src, dst, PAGE_SIZE);
	    }
	    Result::<_, &'static str>::Ok(())
	}).unwrap();
	assert_contents(&loaded_pvec);
    }

    #[test]
    fn test_from_loader_err() {
	let msg = "expected failure";
	let loader = PVec::from_loader(CAPACITY, |_ndx, _dst| Err(msg));

	assert!(loader.is_err());
	assert!(loader.err().unwrap() == msg);
    }
}
