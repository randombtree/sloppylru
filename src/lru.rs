use std::cmp::{Ordering};
use std::convert::{TryInto, TryFrom};
use std::fmt;
use std::ops::{
    Add,
    IndexMut,
    Sub,
    SubAssign
};
use std::cmp::{PartialOrd, PartialEq};

use log::{
    trace,
};

use sled;
use sled::IVec;

use bit_vec::BitVec;

use crate::config::CacheConfig;
use crate::pagedvec::{
    PagedVec,
    PageIter,
    ShadowPages,
};


#[derive(Clone, Copy, Hash, Debug)]
pub(crate) struct Slot(u32);


/// Slot position
impl Slot {
    pub const SLOT_MASK:u32 = (1 << 30) - 1; // 2 high bits unavailable, See LruItem
    pub const NIL:Slot = Slot(u32::MAX & Self::SLOT_MASK);
    pub const MAX:Slot = Slot(Self::NIL.0 - 1);
    pub fn as_key(&self) -> [u8;4] {
	self.0.to_be_bytes()
    }
    pub fn is_nil(&self) -> bool { self.0 == Self::NIL.0 }
}


impl From<Slot> for sled::IVec {
    fn from(slot: Slot) -> Self {
	sled::IVec::from(&slot.as_key())
    }
}

impl From<Slot> for u32 {
    fn from(item: Slot) -> Self { item.0 }
}
impl From<Slot> for u64 {
    fn from(item: Slot) -> Self { item.0 as u64 }
}
impl From<Slot> for usize {
    fn from(item: Slot) -> Self { item.0 as usize }
}

impl From<usize> for Slot {
    fn from(item: usize) -> Self {
	Slot(item as u32)
    }
}

impl From<u32> for Slot {
    fn from(item: u32) -> Self {
	assert!(item < Self::NIL.0);
	Slot(item)
    }
}

impl From<i32> for Slot {
    fn from(item: i32) -> Self {
	assert!(item >= 0);
	Slot(item as u32)
    }
}

impl From<IVec> for Slot {
    fn from(ivec: IVec) -> Self {
	assert!(ivec.len() == 4);
	let mut bytes: [u8; 4] = [0; 4];
	bytes.clone_from_slice(ivec.as_ref());
	Slot(u32::from_be_bytes(bytes))
    }
}

impl From<u64> for Slot {
    fn from(item: u64) -> Self {
	assert!(item < Self::SLOT_MASK as u64);
	Slot(item as u32)
    }
}

impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl PartialEq<u32> for Slot {
    fn eq(&self, other: &u32) -> bool { self.0 == *other }
}

impl PartialEq<usize> for Slot {
    fn eq(&self, other: &usize) -> bool { self.0 == (*other as u32) }
}

impl PartialOrd<u32> for Slot {
    fn partial_cmp(&self, other: &u32) -> Option<Ordering> {
	self.0.partial_cmp(other)
    }
}

impl PartialOrd<Slot> for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
	self.0.partial_cmp(&other.0)
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering { self.0.cmp(&other.0) }
}

impl Eq for Slot {}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	if self.0 == Slot::NIL.0 {
	    write!(f, "<NIL>")
	} else {
	    write!(f, "{}", self.0)
	}
    }
}

struct SlotIterator<'a, I>
where I: IndexMut<usize, Output = LruItem>
{
    items:   &'a mut I,
    first:   Slot,
    current: Slot,
    offset:  i32,
}

impl<'a, I> SlotIterator<'a, I>
where I: IndexMut<usize, Output = LruItem>
{
    /// Construct a SlotIterator
    /// SlotIterator is unsafe, as it's possible to get multiple mutable references to the same items
    /// if you really want to!
    pub unsafe fn new(items: &'a mut I, first: Slot) -> SlotIterator<'a, I> {
	let current = first;
	let offset  = 0;
	SlotIterator {
	    items,
	    first,
	    current,
	    offset,
	}
    }
}

impl<'a, I> Iterator for SlotIterator<'a, I>
where I: IndexMut<usize, Output = LruItem>
{
    type Item = (Slot, &'a mut LruItem);
    fn next(&mut self) -> Option<Self::Item> {
	// Have we gone a full roundabout
	if self.offset > 0 && self.current == self.first {
	    return None
	} else if self.offset < 0 && self.current == self.first {
	    // the reverse has gone around, need fixup to continue forward
	    self.current = self.items[usize::from(self.current)].next();
	    self.offset += 1;
	}
	let current = self.current;
	let current_item = &mut self.items[usize::from(self.current)];
	let next = current_item.next();
	self.current = next;
	self.offset += 1;
	// HACK ALERT:
	// This is a hack to overcome the inability to safely guarantee that
	// multiple mutable ptr:s to the same item aren't passed back
	// In our local usage, this works, but i general this can't be guaranteed!
	let current_ptr = current_item as *mut LruItem;
	// Unsafe: This IS unsafe, and will shoot you in the foot quite easily.
	Some((current, unsafe { std::mem::transmute::<_, &mut LruItem>(current_ptr)} ))
    }

}

impl<'a, I> DoubleEndedIterator for SlotIterator<'a, I>
where I: IndexMut<usize, Output = LruItem>
{
    fn next_back(&mut self) -> Option<Self::Item> {
	if self.offset < 0 && self.current == self.first {
	    return None
	} else if self.offset > 0 && self.current == self.first {
	    // The forward iterator has gone around; need fixup
	    self.current = self.items[usize::from(self.current)].prev();
	    self.offset -= 1;
	}
	let current = self.current;
	let current_item = &mut self.items[usize::from(self.current)];
	let prev = current_item.prev();
	self.current = prev;
	self.offset -= 1;
	// HACK ALERT: Read the note in next() above
	let current_ptr = current_item as *mut LruItem;
	// Unsafe: ditto!
	Some((current, unsafe { std::mem::transmute::<_, &mut LruItem>(current_ptr)} ))
    }
}


/// 4-bit level
/// TODO: Lot's of asserts for over & underflow
#[derive(Copy, Clone, Debug)]
pub(crate) struct Level(u8);

impl From<u32> for Level {
    fn from(v: u32) -> Level {
	assert!(v <= LruItem::LEVEL_MAX as u32);
	Level(v as u8)
    }
}

impl From<usize> for Level {
    fn from(v: usize) -> Level {
	assert!(v <= LruItem::LEVEL_MAX as usize);
	Level(v as u8)
    }
}

impl From<Level> for u64 {
    fn from(l: Level) -> u64 { l.0 as u64 }
}

impl From<Level> for usize {
    fn from(l: Level) -> usize { l.0 as usize }
}

impl Add<Level> for usize {
    type Output = usize;
    fn add(self, l: Level) -> Self::Output {
	self + (l.0 as usize)
    }
}

impl Add<&Level> for usize {
    type Output = usize;
    fn add(self, l: &Level) -> Self::Output {
	self + (l.0 as usize)
    }
}

impl Add<usize> for Level {
    type Output = Level;
    fn add(self, n: usize) -> Self::Output {
	Level(self.0 + u8::try_from(n).unwrap())
    }
}

impl Sub<u32> for Level {
    type Output = Level;
    fn sub(self, n: u32) -> Self::Output {
	assert!(n <= self.0 as u32);
	Level(self.0 - u8::try_from(n).unwrap())
    }
}

impl SubAssign<u32> for Level {
    fn sub_assign(&mut self, rhs: u32) {
	assert!(rhs <= self.0 as u32);
	self.0 -= u8::try_from(rhs).unwrap();
    }
}

impl PartialEq for Level {
    fn eq(&self, other: &Level) -> bool { self.0 == other.0 }
}

impl PartialOrd for Level {
    fn partial_cmp(&self, other: &Level) -> Option<Ordering> {
	self.0.partial_cmp(&other.0)
    }
}


/// A packed triplet of slot pointers and level (30 + 30 + 4)
#[derive(Copy, Clone, PartialEq)]
pub(crate) struct LruItem(u64);


impl LruItem {
    const NEXT_SHIFT: u64    = 30;
    const PREV_MASK: u64     = Slot::SLOT_MASK as u64;
    const NEXT_MASK: u64     = (Slot::SLOT_MASK as u64) << Self::NEXT_SHIFT;
    const LEVEL_SHIFT: u64   = 2 * Self::NEXT_SHIFT;
    const LEVEL_BITS: u64    = 4;
    const LEVEL_MASK: u64    = ((1 << Self::LEVEL_BITS) - 1) << Self::LEVEL_SHIFT;
    const LEVEL_MAX: u8      = ((1 << Self::LEVEL_BITS) - 1) as u8;
    const DETACHED: u64   = !0u64;

    pub(crate) fn new(prev: Slot, next: Slot, level: Level) -> Self {
	let prev: u64 = prev.into();
	let next: u64 = u64::from(next) << Self::NEXT_SHIFT;
	let level: u64 = u64::from(level) << Self::LEVEL_SHIFT;
	LruItem(level | next | prev)
    }

    pub(crate) fn prev(&self) -> Slot {
	Slot::from(self.0 & Slot::SLOT_MASK as u64)
    }

    pub(crate) fn next(&self) -> Slot {
	Slot::from((self.0 >> Self::NEXT_SHIFT) & Slot::SLOT_MASK as u64)
    }

    pub(crate) fn set_prev(&mut self, slot: Slot) {
	self.0 = u64::from(slot) | (self.0 & !Self::PREV_MASK)
    }

    pub(crate) fn set_next(&mut self, slot: Slot) {
	let next = u64::from(slot) << Self::NEXT_SHIFT;
	self.0 = next | (self.0 & !Self::NEXT_MASK);
    }

    pub(crate) fn level(&self) -> Level {
	Level((self.0 >> Self::LEVEL_SHIFT) as u8)
    }

    pub(crate) fn set_level(&mut self, level: Level) {
	let level = (level.0 as u64) << Self::LEVEL_SHIFT;
	self.0 = level | (self.0 & !Self::LEVEL_MASK);
    }

    /// Set this item to a detched state; this is mostly a debug aid
    /// to track invalid modifications of the node.
    pub(crate) fn set_detached(&mut self) {
	self.0 = Self::DETACHED;
    }

    pub(crate) fn is_detached(&mut self) -> bool {
	self.0 == Self::DETACHED
    }
}


#[derive(Clone,Copy, Debug, PartialEq)]
struct LevelState {
    // This is a soft max, only enforced when balancing
    max_size: usize,
    allocated: usize,
}


// Iterator over a specific LRU level
pub(crate) struct LevelIterator<'a, const N: usize>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    lru: &'a LruArray<N>,
    head_slot: Slot,
    current: Option<Slot>,
}


impl<'a, const N: usize> LevelIterator<'a, N>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    fn new(lru: &'a LruArray<N>, level: Level) -> LevelIterator<'a, N> {
	let head_slot = lru.level_head(level).into();
	let current = Some(head_slot);
	LevelIterator {
	    lru,
	    head_slot,
	    current,
	}
    }
}


impl<'a, const N: usize> Iterator for LevelIterator<'a, N>
        where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    type Item = Slot;
    fn next(&mut self) -> Option<Self::Item> {
	let current = self.current?;

	let current = self.lru.items[usize::from(current)].next();
	let current = {
	    if current == self.head_slot {
		// Gone full circle
		None
	    } else {
		Some(current)
	    }
	};
	self.current = current;
	current
    }
}


#[cfg(test)]
pub(crate) struct LruIterator<'a, const N: usize>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    index: usize,
    lru: &'a LruArray<N>,
}

#[cfg(test)]
impl<'a, const N: usize> LruIterator<'a, N>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    fn new(lru: &'a LruArray<N>) -> LruIterator<'a, N> {
	LruIterator {
	    index: 0,
	    lru,
	}
    }
}

#[cfg(test)]
impl<'a, const N: usize> Iterator for LruIterator<'a, N>
        where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    type Item = &'a LruItem;
    fn next(&mut self) -> Option<Self::Item> {
	self.lru.inspect(self.index).and_then(|i| {
	    self.index += 1;
	    Some(i)
	})
    }
}

/// Iterator over raw LRU page contents.
pub struct LruPageIterator<'a, const N: usize>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    page_iter: PageIter<'a, LruItem, LRU_PAGE_SIZE>,
    phantom: std::marker::PhantomData<&'a LruArray<N>>,
}

impl<'a, const N: usize> LruPageIterator<'a, N>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    fn new(page_iter: PageIter<'a, LruItem, LRU_PAGE_SIZE>) -> LruPageIterator<'a, N> {
	LruPageIterator {
	    page_iter,
	    phantom: std::marker::PhantomData,
	}
    }
}

impl<'a, const N: usize> Iterator for LruPageIterator<'a, N>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    type Item = &'a [u8; LRU_PAGE_SIZE];
    fn next(&mut self) -> Option<Self::Item> {
	self.page_iter.next()
    }
}


/// Track changed elements in LRUarray (with page granularity).
/// Todo: Evaluate performance hit of using HashSet for tracking mask
///       NB: change_mask eats quite a bit of memory, but it's very cache-line
///           and branch-predictor efficient when modified.
struct LruChangeTracker<const N: usize> {
    change_mask: BitVec,
    change_set:  Vec<usize>,
}

impl<const N: usize> LruChangeTracker<N> {
    const ITEMS_PER_PAGE: usize = LRU_PAGE_SIZE / std::mem::size_of::<LruItem>();
    fn new(pages: usize) -> LruChangeTracker<N> {
	let mut change_mask = BitVec::with_capacity(pages);
	change_mask.grow(pages, false);
	LruChangeTracker {
	    change_mask,
	    change_set: Vec::new(),
	}
    }

    /// Add index `i` to change-set.
    fn add(&mut self, i: usize) {
	let pg_num = i / Self::ITEMS_PER_PAGE;
	match self.change_mask.get(pg_num) {
	    Some(v) if !v => {
		//trace!("Changed {} page {}", i, pg_num);
		self.change_mask.set(pg_num, true);
		self.change_set.push(pg_num);
	    },
	    _ => (),
	}
    }

    /// Convinience function to add many changes, see: `add`.
    fn add_many(&mut self, indexes: &[usize]) {
	for i in indexes {
	    self.add(*i);
	}
    }

    /// Reset changes to 0, returning the current changes in a sorted list.
    fn reset(&mut self) -> Vec<usize> {
	self.change_mask.clear();
	// Anticipate capacity from previous experiences
	let mut ret = Vec::with_capacity(self.change_set.capacity());
	std::mem::swap(&mut ret, &mut self.change_set);
	ret.sort();
	ret
    }
}


/// On disk LRU meta header
#[derive(Clone, Copy)]
#[repr(packed)]
struct LruMetaHeaderBytes<const N: usize>
where [(); N + 2]: Sized
{
    size:            usize,
    free_slack:      usize,
    free_hysteresis: usize,
    levels:          [LevelState; N + 2],
}


/// Create shorthand name for a list item
macro_rules! name_index {
    ($name:ident, $list: expr, $ndx:ident) => {
	macro_rules! $name {
	    () => { ($list[$ndx.into()]) }
	}
    }
}


/// Gets two (safe) mutable references to list items (consuming list reference)
macro_rules! take_two_refs {
    (@sorted, $list:expr, $first:ident, $second:ident) => {
	{
	    assert!($first < $second);
	    let (a, b) = $list.split_at_mut($second);
	    let second = &mut b[0];
	    let (_a, b) = a.split_at_mut($first);
	    let first  = &mut b[0];
	    (first, second)
	}
    };
    ($list:expr, $first_ndx:expr, $second_ndx:expr) => {
	{
	    let (first, second);
	    let (first_ndx, second_ndx) = ($first_ndx, $second_ndx);
	    if $first_ndx < $second_ndx {
		(first, second) = take_two_refs!(@sorted, $list, first_ndx, second_ndx);
	    } else {
		(second, first) = take_two_refs!(@sorted, $list, second_ndx, first_ndx);
	    }
	    (first, second)
	}
    }
}


/// Page size used when writing LRU to stable storage.
pub const LRU_PAGE_SIZE: usize = 4096;

/// How many items are stored per LRU page.
/// For testing and optimization of page usage
pub const ITEMS_PER_PAGE: usize = LRU_PAGE_SIZE / std::mem::size_of::<LruItem>();

pub(crate) struct LruArray<const N: usize>
where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    size:        usize,
    free_slack:  usize,
    free_hysteresis: usize,
    items:       PagedVec<LruItem, LRU_PAGE_SIZE>,   // TODO: Configurable page size
    levels:      [LevelState;N + 2],
    changes:     LruChangeTracker<N>,
}

impl<const N: usize> LruArray<N>
where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
    [(); N + 2]: Sized
{
    const LEVEL_FREE: Level       = Level(0u8);  // Free item list
    const LEVEL_FREE_ALLOC: Level = Level(1u8);  // Items queued to be free'd
    const LEVEL_USER_START: Level = Level(2u8);  // Start of regular levels
    const LEVELS: usize           = N + 2;

    /// Header size
    const META_BYTES: usize       = std::mem::size_of::<LruMetaHeaderBytes<N>>();
    const ITEMS_PER_PAGE: usize   = LRU_PAGE_SIZE / std::mem::size_of::<LruItem>();

    pub fn new(config: &CacheConfig<N>) -> Self {
	let free_slack = config.free_slack;
	let free_hysteresis = ((config.free_hysteresis_pct as f64) / 100.0 * free_slack as f64).floor() as usize;
	let size = config.blocks;
	let total = size + Self::LEVELS;
	// Vec with room for all items + level heads
	let items = PagedVec::new(total, move |i| {
	    let (prev, next) = {
		if i == (size + Self::LEVEL_FREE.0 as usize) {
		    (size - 1, 0)
		} else if i > size {
		    // All other levels are empty, thus loop-back
		    (i, i)
		} else if i == 0 {
		    (size + Self::LEVEL_FREE.0 as usize, 1)
		} else if i == size - 1 {
		    // This is just to be defencive if level_free is ever defined
		    // in another level
		    (i - 1, size + Self::LEVEL_FREE.0 as usize)
		} else {
		    // In between is just linking predecessor and sucessor
		    (i - 1, i + 1)
		}
	    };
	    // First and last items are linked to the free-list head
	    LruItem::new(Slot::from(prev), Slot::from(next), Self::LEVEL_FREE)
	});

	let levels: [LevelState; N + 2] = (0..(N + 2)).map(|i| {
	    let (max_size, allocated) = {

		if i == Self::LEVEL_FREE.into() {
		    // Free level gets all the blocks
		    (size, size)
		} else if i == Self::LEVEL_FREE_ALLOC.into() {
		    // Free, but allocated gets no limit (not that it matters)
		    (size, 0)
		} else {
		    // User level + free alloc level
		    // TODO: Consolidate block calculation between config and here
		    (((config.level_pct[i - 2] as f64) / 100.0 * size as f64).floor() as usize ,
		     0)
		}
	    };
	    LevelState {
		max_size,
		allocated,
	    }
	}).collect::<Vec<LevelState>>().try_into().unwrap();

	let pg_count = items.pages();
	LruArray {
	    size,
	    free_slack,
	    free_hysteresis,
	    items,
	    levels,
	    changes: LruChangeTracker::new(pg_count),
	}
    }

    /// Load state from the meta-block and the content pages through the loader.
    pub fn from_loader<F, E>(meta: &[u8], loader: F) -> Result<Self, E>
    where
	F: FnMut(usize, &mut [u8; LRU_PAGE_SIZE]) -> Result<(), E>
    {
	assert!(meta.len() == Self::META_BYTES);
	// Meta layout (also see LruMetaHeaderBytes):
	// - size                 => SIZE_BYTES
	// - free_slack          \
	// - free_hysteresis      => SLACK_BYTES
	// - [LevelState; N + 2]  => LEVEL_BYTES
	let base_ptr = meta.as_ptr();
	// Unsafe: Guarded by META_BYTES check
	let LruMetaHeaderBytes { size, free_slack, free_hysteresis, levels } = unsafe {
	    *(base_ptr as *const LruMetaHeaderBytes<N>)
	};
	let total = size + Self::LEVELS;

	trace!("Loading LRU of size {}", size);
	let items = PagedVec::<_, LRU_PAGE_SIZE>::from_loader(total, loader)?;
	let pg_count = items.pages();
	Ok(LruArray {
	    size,
	    free_slack,
	    free_hysteresis,
	    items,
	    levels,
	    changes: LruChangeTracker::new(pg_count),
	})
    }

    /// Do Balance & return recycled slot numbers
    /// These slots must not be used after the call!
    pub fn maybe_gc(&mut self) -> Option<Vec<Slot>> {
	let free_slots = self.levels[usize::from(Self::LEVEL_FREE)].allocated;
	if free_slots < self.free_hysteresis {
	    let free_allocated = self.levels[usize::from(Self::LEVEL_FREE_ALLOC)].allocated;
	    let total_free = free_slots + free_allocated;
	    if  total_free < self.free_slack {
		// We are really out of space, need to balance to reclaim some slots
		let balanced = self.balance();
		// The balance should always result in at least free_slack total items
		assert!(balanced >= self.free_slack - total_free);
	    }
	    let move_count = self.free_slack - free_slots;
	    // Adjust book-keeping.
	    self.levels[usize::from(Self::LEVEL_FREE)].allocated += move_count;
	    self.levels[usize::from(Self::LEVEL_FREE_ALLOC)].allocated -= move_count;

	    let mut items = self.items.get_locked_mut();

	    let t_head_ndx = Slot::from(self.size + Self::LEVEL_FREE);
	    let s_head_ndx = Slot::from(self.size + Self::LEVEL_FREE_ALLOC);
	    name_index!(t_head, items, t_head_ndx);
	    name_index!(s_head, items, s_head_ndx);

	    let s_last_ndx  = s_head!().prev();
	    let t_first_ndx = t_head!().next();  // Can be head!
	    name_index!(s_last, items, s_last_ndx);
	    name_index!(t_first, items, t_first_ndx);
	    // Move a chain from source tail to destination, this way we don't
	    // need to modify that many items
	    // t_head    / a, b, c \  -> t_first_ndx
	    // s_head <- \ a, b, c /
	    // First gather the changed indexes, starting in reverse order.
	    // Unsafe: Not changing direction of iterator, so SlotIterator is safe.
	    let indexes: Vec<Slot> = unsafe { SlotIterator::new(&mut items, s_last_ndx) }
		.rev().take(move_count)
		.map(|(slot, item)| {
		    item.set_level(Self::LEVEL_FREE);
		    slot
		}).collect();

	    let s_first_ndx = Slot::from(indexes[indexes.len() - 1]);
	    name_index!(s_first, items, s_first_ndx);
	    // The remaining item, is the previous one (remember, we walked backwards)
	    let s_remaining_ndx = s_first!().prev();
	    name_index!(s_remaining, items, s_remaining_ndx);

	    // Link in the first item
	    s_first!().set_prev(t_head_ndx);
	    t_head!().set_next(s_first_ndx);

	    // Link in the last item
	    s_last!().set_next(t_first_ndx);
	    t_first!().set_prev(s_last_ndx);

	    // Heal the source list, linking in the remaining item
	    s_last!().set_next(t_first_ndx);
	    s_head!().set_prev(s_remaining_ndx);
	    s_remaining!().set_next(s_head_ndx);

	    let mut modified: Vec<usize> = indexes.iter().map(|slot| (*slot).into()).collect();
	    // We also modified the heads and the link in/out items.
	    modified.push(s_head_ndx.into());
	    modified.push(t_head_ndx.into());
	    modified.push(t_first_ndx.into());
	    modified.push(s_remaining_ndx.into());

	    self.changes.add_many(modified.as_slice());
	    Some(indexes)
	} else {
	    None
	}
    }

    /// Balance the levels, i.e. move items down a notch if they overflow the level allowance
    /// Returns count of items returned to the free alloc list (i.e. GC is required).
    pub(crate) fn balance(&mut self) -> usize {
	trace!("Starting balance");
	// Balance user levels
	let mut balanced: usize = 0;
	// Work down from highest level, donwards
	for ndx in (usize::from(Self::LEVEL_USER_START)..Self::LEVELS).rev() {
	    balanced = 0;
	    let (t_level, s_level) = take_two_refs!(self.levels, ndx - 1, ndx);

	    trace!("Level {} allocated = {} (max: {})",
		   ndx, s_level.allocated, s_level.max_size);
	    // Level is balanced, save a few cycles
	    if s_level.allocated <= s_level.max_size {
		continue;
	    }

	    // Move last item "down" in level hierarchy
	    let s_head_ndx     = self.size + ndx;
	    let t_head_ndx     = self.size + ndx - 1;

	    name_index!(t_head, self.items, t_head_ndx);

	    let mut last_prev_ndx: Option<usize> = None;

	    while s_level.allocated > s_level.max_size {
		s_level.allocated -= 1;
		t_level.allocated += 1;
		balanced += 1;
		// Take from the tail, give to the head
		let [t_head, s_head] = self.items.take_refs_mut([t_head_ndx, s_head_ndx]);
		let item_ndx: usize  = s_head.prev().into();      // The index we are moving one level down
		let first_ndx: usize = t_head.next().into();      // The previously first item on target

		trace!("Moving item {}", item_ndx);

		// Can't take t_head here because it might be same as "first"
		let [item,
		     first,
		     s_head] = self.items.take_refs_mut_unsorted([item_ndx, first_ndx, s_head_ndx]);
		let new_prev_ndx: usize = item.prev().into();

		item.set_next(first_ndx.into());
		item.set_prev(t_head_ndx.into());
		item.set_level((ndx - 1).into());

		first.set_prev(item_ndx.into());
		s_head.set_prev(new_prev_ndx.into());
		// Update heads
		t_head!().set_next(item_ndx.into());
		// Record the changes (heads recorded separately)
		self.changes.add_many(&[item_ndx, first_ndx]);

		// NB: Fixup 'source tail' later on when writing to array, as we only need to do it once
		// after all items are balanced
		last_prev_ndx = Some(new_prev_ndx);
	    }
	    // As noted above, the last item remaining on the source level must have its next ptr fixed
	    let last_ndx = last_prev_ndx.unwrap();
	    self.items[last_ndx].set_next(s_head_ndx.into());
	    self.changes.add_many(&[last_ndx, s_head_ndx, t_head_ndx]);
	    trace!("Balanced {} items on level {}", balanced, ndx);
	}
	balanced
    }

    fn level_head(&self, level: Level) -> usize
    {
	self.size + level
    }

    /// Get a free slot from level
    pub fn get(&mut self, level: usize) -> Option<Slot>
    {
	assert!(level < N);
	let level_free: &mut LevelState = &mut self.levels[usize::from(Self::LEVEL_FREE)];
	if level_free.allocated == 0 {
	    return None;
	}
	level_free.allocated -= 1;

	let user_level    = Self::LEVEL_USER_START + level;
	let free_head_ndx = self.level_head(Self::LEVEL_FREE);
	let free_head     = self.items[free_head_ndx];
	let slot          = free_head.prev();                   // <- Our slot
	let slot_ndx: usize = slot.into();
	let slot_item     = self.items[usize::from(slot)];
	let new_last_ndx:usize  = slot_item.prev().into();

	// Update "new last" (if allocated == 0, this is the head)
	self.items[new_last_ndx]
	    .set_next(free_head_ndx.into());

	// Update free head
	self.items[free_head_ndx]
	    .set_prev(new_last_ndx.into());

	// Update user level head
	let level_head_ndx = self.level_head(user_level);
	let level_head = self.items[level_head_ndx];
	let next_slot_ndx: usize = level_head.next().into();
	self.items[next_slot_ndx]
	    .set_prev(slot);
	let  [slot_item, level_head] = self.items.take_refs_mut([slot_ndx, level_head_ndx]);
	level_head.set_next(slot);
	slot_item.set_next(next_slot_ndx.into());
	slot_item.set_prev(level_head_ndx.into());
	slot_item.set_level(user_level.try_into().unwrap()); // Compile time constraint makes this always work!
	// And update book-keeping
	self.levels[usize::from(user_level)].allocated += 1;
	// And what indexes were touched
	self.changes.add_many(&[free_head_ndx, level_head_ndx, slot_ndx, new_last_ndx, next_slot_ndx]);
	Some(slot)
    }

    /// Promote slot - it moves to the top of its level
    /// @level Optionally specify a different level to move into (this can also
    ///        de-promote the slot)
    pub fn promote(&mut self, slot: Slot, level: Option<usize>)
    {
	let item_ndx = usize::from(slot);
	name_index!(item, self.items, item_ndx);
	let s_level = item!().level();
	// What level are we promoting to?
	let t_level = {
	    if let Some(level) = level {
		Self::LEVEL_USER_START + level
	    } else {
		s_level
	    }
	};
	trace!("Promoting {} from {:?} to {:?}", item_ndx, s_level, t_level);
	assert!(usize::from(t_level) < Self::LEVELS);

	self.levels[usize::from(s_level)].allocated -= 1;
	self.levels[usize::from(t_level)].allocated += 1;
	// Installing item at target head
	let t_head_ndx = self.level_head(t_level);
	let [item, t_head] = self.items.take_refs_mut([item_ndx, t_head_ndx]);
	let old_top_ndx: usize = t_head.next().into();
	if old_top_ndx == item_ndx {
	    trace!("Item already at top of level");
	    return;
	}

	// Store old indexes for item
	let prev_ndx: usize = item.prev().into();
	let next_ndx: usize = item.next().into();

	// And then insert item at H <-> I <-> O
	item.set_prev(t_head_ndx.into());
	item.set_next(old_top_ndx.into());
	item.set_level(t_level);
	t_head.set_next(item_ndx.into());
	name_index!(old_top, self.items, old_top_ndx);
	old_top!().set_prev(slot);

	// Patch hole P <-> (I) <-> N
	name_index!(prev, self.items, prev_ndx);
	name_index!(next, self.items, next_ndx);
	prev!().set_next(next_ndx.into());
	next!().set_prev(prev_ndx.into());
	self.changes.add_many(&[item_ndx, prev_ndx, next_ndx, t_head_ndx, old_top_ndx]);
    }

    // Get a boxed slice with the meta bytes
    fn get_meta(&self) -> Box<[u8]> {
	// A bit inefficient allocation, but compiler might some day be intelligent
	// enough to optimize this out? :)
	let meta_bytes: Vec<u8> = vec![0; Self::META_BYTES];
	let mut meta_bytes = meta_bytes.into_boxed_slice();
	// Unsafe: The meta header is always == META_BYTES
	let meta = unsafe {
	    &mut *((*meta_bytes).as_mut_ptr() as *mut LruMetaHeaderBytes<N>)
	};
	meta.size = self.size;
	meta.free_slack = self.free_slack;
	meta.free_hysteresis = self.free_hysteresis;
	meta.levels = self.levels;

	meta_bytes
    }

    /// Save the whole lru; This also counts as a checkpoint..
    pub fn save<'a>(&'a mut self) -> (Box<[u8]>, LruPageIterator<'a, N>) {
	let pvec_iter = self.items.page_iter();
	let iter = LruPageIterator::new(pvec_iter);
	let meta = self.get_meta();
	self.changes.reset();
	(meta, iter)
    }

    /// Return the changed pages since last checkpoint; If `full_checkpoint` is
    /// set, all pages are returned. Initialization isn't counted as a "change"
    /// (for obvious reasons), so before checkpointing you should save the whole
    /// lru with `save()`.
    ///  This returns the current immutable state that can be written to disk.
    /// TODO: Currently not endian-safe, i.e. saved state can't be read on a computer
    ///       with a differing endian-cpu; e.g. x86 <-> SPARC
    pub fn checkpoint(&mut self) -> (Box<[u8]>, ShadowPages<LRU_PAGE_SIZE>) {
	//let level_states;
	let changes = self.changes.reset();
	trace!("{} changes", changes.len());
	// FIXME: API thinko: we are indexing with page granularity, but shadow
	// takes the changes as index grandularity,
	// for now, just translate to per page indexes until fixex with
	// a better API.
	let changes: Vec<usize> = changes.iter().map(|page| page * Self::ITEMS_PER_PAGE).collect();
	let shadow = self.items.take_shadow_pages(changes.as_slice());
	let meta = self.get_meta();
	(meta, shadow)
    }

    /// Inspect LruItem at specific index, only for testing and validation
    #[cfg(test)]
    pub(crate) fn inspect(&self, item: usize) -> Option<&LruItem> {
	self.items.get(item)
    }

    /// Iterator over all LruItems, including the heads
    /// For testing purposes; validating content.
    #[cfg(test)]
    pub(crate) fn iter<'a>(&'a self) -> LruIterator<'a, N> {
	LruIterator::new(self)
    }

    #[cfg(test)]
    pub fn level_iter<'a, const L: usize>(&'a self) -> LevelIterator<'a, N>
    where [(); N - L - 1]: Sized,
    {
	let user_level = L + 2;
	LevelIterator::new(self, user_level.into())
    }

    #[cfg(test)]
    fn get_slots<const L: usize, T>(&mut self, n: u32) -> T
	where
	[(); N - L - 1]: Sized,
	T: FromIterator<Slot>,
    {
	(0..n).map(|_| self.get(L).unwrap()).collect()
    }
}

impl<const N: usize> PartialEq for LruArray<N>
where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    fn eq(&self, other: &Self) -> bool {
	let size = self.size == other.size;
	let slack = self.free_slack == other.free_slack && self.free_hysteresis == other.free_hysteresis;
	let levels = self.levels.iter().zip(other.levels.iter()).all(|(a, b)| a == b);
	let items = self.items.iter().zip(other.items.iter()).all(|(a,b)| a == b);
	if !size {
	    trace!("Size differs");
	}
	if !slack {
	    trace!("Slack differs");
	}
	if !levels {
	    trace!("Levels differ");
	}
	if !items {
	    trace!("Items differ");
	}
	size && slack && levels && items
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{VecDeque, HashSet};
    use std::collections::hash_map::RandomState;
    use std::fs::File;
    use std::io::{Read, Write, Seek, SeekFrom};

    use log::{ trace };
    use test_log::test;

    use super::{LruArray, Slot};
    use crate::config::CacheConfig;
    use crate::test::TestPath;

    // TODO: Macro for generating tests for different configurations
    const L1: usize = 0;
    const L2: usize = 1;
    const L3: usize = 2;
    const L4: usize = 3;
    const LEVELS: usize = 3;
    // LRU item count, fill 10 pages to test out paging (the 11:th will have the heads)
    const PAGES: usize = 10;
    const TOTAL_PAGES: usize = PAGES + 1;
    const SIZE: usize = PAGES * super::LRU_PAGE_SIZE / 8;
    const TOTAL_SIZE: usize = SIZE + LruArray::<LEVELS>::LEVELS;
    const ITEMS_PER_PAGE: usize = super::LRU_PAGE_SIZE / 8;

    // Compare
    fn compare_level<'a, A, B, C, ITER>(checklist: A, mut level_iterator: B) -> bool
    where
	A: IntoIterator<Item = &'a C, IntoIter = ITER>,
	B: Iterator<Item = C>,
	C: PartialEq + std::fmt::Display + 'a,
	ITER: DoubleEndedIterator<Item = &'a C>
    {
	let list_iter = checklist.into_iter().rev();
	for (ndx, item) in list_iter.enumerate() {
	    if let Some(other) = level_iterator.next() {
		if *item != other {
		    trace!("Mismatch at index {}: {} != {}", ndx, *item, other);
		    return false;
		}
	    } else {
		// Lengths differ
		trace!("list iterator is longer?");
		return false;
	    }
	}

	// lengths can differ
	match level_iterator.next() {
	    Some(_) => {trace!("level iterator is longer?"); false},
	    _ => true,
	}
    }

    /// Save lru checkpoint data to file
    fn save_checkpoint(file: &mut File, lru: &mut LruArray<LEVELS>) -> Box<[u8]>{
	let (meta, pages) = lru.checkpoint();
	trace!("Pages to write: {}", pages.as_slice().len());
	for (pg_num, data) in pages.as_slice().iter() {
	    trace!("Writing page {}: {} - {}", pg_num, pg_num * ITEMS_PER_PAGE, (pg_num + 1) * ITEMS_PER_PAGE - 1);
	    file.seek(SeekFrom::Start(u64::try_from(pg_num * super::LRU_PAGE_SIZE).unwrap())).unwrap();
	    file.write(*data).unwrap();
	}
	meta
    }

    /// Construct lru from meta and saved pages on disk
    fn lru_from_loader(meta: &[u8], save_file: &mut File) -> LruArray<LEVELS> {
	save_file.seek(SeekFrom::Start(0)).unwrap();
	// Expect that the pages are loaded in order
	let mut expect_ndx: usize = 0;

	LruArray::<LEVELS>::from_loader(&meta, |ndx, buf: &mut [u8; super::LRU_PAGE_SIZE]| {
	    assert!(ndx == expect_ndx, "Not loading pages in order!?");
	    expect_ndx += 1;
	    // No need to seek; going in order
	    save_file.read(buf).and_then(|bytes| {
		assert!(bytes == super::LRU_PAGE_SIZE);
		Ok(())
	    })
	}).unwrap()
    }

    #[test]
    fn test_lru() {
	let tmp_path = TestPath::new("lru_test_lru").unwrap();

	let config = CacheConfig::<LEVELS>::default(SIZE);
	let mut lru = LruArray::new(&config);
	let mut l1_slots = lru.get_slots::<L1, VecDeque<Slot>>(10);
	let mut l2_slots = lru.get_slots::<L2, VecDeque<Slot>>(10);
	let mut l3_slots = lru.get_slots::<L3, VecDeque<Slot>>(10);
	// This must fail at compile time, however how to test it!?
	//let mut l4_slots = lru.get_slots::<L4>(10);

	println!("LEVEL 1 {:?}", l1_slots);
	println!("LEVEL 2 {:?}", l2_slots);
	println!("LEVEL 3 {:?}", l3_slots);
	macro_rules! compare_level {
	    () => {
		assert!(compare_level(&l1_slots, lru.level_iter::<L1>()));
		assert!(compare_level(&l2_slots, lru.level_iter::<L2>()));
		assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));
	    }
	}
	compare_level!();


	// Now fill up all levels (to test availability and, later, the balancer
	let r1 = (config.level_pct[L1] as f64 / 100.0 * config.blocks as f64).floor() as usize - 10;
	let r2 = (config.level_pct[L2] as f64 / 100.0 * config.blocks as f64).floor() as usize - 10;
	let r3 = (config.level_pct[L3] as f64 / 100.0 * config.blocks as f64).floor() as usize - 10;
	let lists = [&mut l1_slots, &mut l2_slots, &mut l3_slots];

	for (ndx, count) in [r1, r2, r3].iter().enumerate() {
	    for _ in 0..*count {
		lists[ndx].push_back(lru.get(ndx).unwrap());
	    }
	}

	// TODO: Test slot uniqueness
	compare_level!();

	// Try saving and restoring contents

	let checkpoint = lru.save();
	let save_file_name = tmp_path.as_ref().join("saved_lru");
	let mut save_file = File::options()
	    .create_new(true)
	    .read(true)
	    .write(true)
	    .open(save_file_name).unwrap();

	// Write pages and levels to file
	for page in checkpoint.1 {
	    save_file.write_all(page).unwrap();
	}
	// Lazy, stuff meta in array
	let meta = checkpoint.0;

	let lru2 = lru_from_loader(&meta, &mut save_file);

	assert!(compare_level(&l1_slots, lru2.level_iter::<L1>()));
	assert!(compare_level(&l2_slots, lru2.level_iter::<L2>()));
	assert!(compare_level(&l3_slots, lru2.level_iter::<L3>()));
	assert!(lru == lru2);
	drop(lru2);
	// Checkpoint lru and re-load from checkpoint and validate
	macro_rules! test_checkpoint {
	    () => {
		let meta = save_checkpoint(&mut save_file, &mut lru);
		let lru2 = lru_from_loader(&meta, &mut save_file);
		if lru != lru2 {
		    for (ndx, (item1, item2)) in lru.iter().zip(lru2.iter()).enumerate() {
			if item1 != item2 {
			    trace!("Items differ at {}", ndx);
			    assert!(false);
			}
		    }
		}
	    }
	}
	test_checkpoint!();

	// ^ /save
	// Now we need to see that checkpointing saves everything
	// Still using the same file for checkpoints so that we can restore from the loader


	// It should still be in great balance
	let balanced = lru.balance();
	assert!(balanced == 0);
	// But now it will overflow; no problems for now however
	l3_slots.push_back(lru.get(L3).unwrap());
	compare_level!();
	// Balance should put one item to the cleaner
	assert!(lru.balance() == 1);

	test_checkpoint!();

	// And now should be in balance again
	assert!(lru.balance() == 0);

	// And now all levels should have changed (first item moved down on each level)
	// (i.e. last in comparator list)
	l2_slots.push_back(l3_slots.pop_front().unwrap());
	l1_slots.push_back(l2_slots.pop_front().unwrap());
	// Save it for later checks
	let l1_drop = l1_slots.pop_front().unwrap(); // <- And this should be in alloc_free list
	compare_level!();

	test_checkpoint!();

	// Promote the previously "downgraded" items;
	let l2_promote = l1_slots.pop_back().unwrap();
	let l3_promote = l2_slots.pop_back().unwrap();
	l2_slots.push_back(l2_promote);
	l3_slots.push_back(l3_promote);
	lru.promote(l2_promote, Some(1));
	lru.promote(l3_promote, Some(2));

	compare_level!();

	test_checkpoint!();

	// Promote again; This must also be checked for!
	lru.promote(l2_promote, Some(1));
	compare_level!();

	test_checkpoint!();

	// Test GC, first fill it up
	let mut items = 0;
	while let Some(slot) = lru.get(L3) {
	    items += 1;
	    l3_slots.push_back(slot);
	}
	compare_level!();
	trace!("Filled LRU with {} items", items);
	test_checkpoint!();

	let gc_list = lru.maybe_gc();

	assert!(gc_list.is_some());
	// Test that we got the "last" items from FREE_ALLOC:
	// This is a bit fragile, might blow up if logic optimizations are done,
	// So needs to be re-worked whenever that happens
	let mut gc_list = gc_list.unwrap();
	let mut last_l1: Vec<Slot> = l1_slots.iter().take(gc_list.len() - 1).map(|r| *r).collect();
	// We had a balance test earlier that moved one item to the FREE_ALLOC list, need to include
	// it here:
	last_l1.push(l1_drop);
	gc_list.sort();
	last_l1.sort();
	for (n, (gc, l1)) in gc_list.iter().zip(last_l1.iter()).enumerate() {
	    assert!(gc == l1, "GC didn't collect the expected slot {} at {}, found = {}",
		    l1, n, gc);
	 }

	test_checkpoint!();

	// Finally, we should be able to get all the items GC:d back again
	let gc_count = gc_list.len();
	let mut gc_set: HashSet<Slot, RandomState> = HashSet::from_iter(gc_list.into_iter());
	let mut counter = 0;
	while let Some(slot) = lru.get(L3) {
	    assert!(gc_set.remove(&slot), "Slot {} not found in GC list? Get #{}", slot, counter);
	    counter += 1;
	}
	// We know all returned items were in gc_set
	assert!(counter == gc_count, "Get didn't return all GC:d items, {} remaining of {}",
		gc_count - counter, gc_count);

	test_checkpoint!();
    }
}
