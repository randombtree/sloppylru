use std::cmp::{Ordering};
use std::convert::{TryInto, TryFrom};
use std::fmt;
use std::ops::{Add, Sub, SubAssign};
use std::cmp::{PartialOrd, PartialEq};

use log::{
    trace,
};

use sled;
use sled::IVec;

use crate::config::CacheConfig;


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
#[derive(Copy, Clone)]
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


#[derive(Clone,Copy, Debug)]
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


pub(crate) struct LruIterator<'a, const N: usize>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    lru: &'a LruArray<N>,
    level: Level,
    iter: LevelIterator<'a, N>,
}


impl<'a, const N: usize> LruIterator<'a, N>
    where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    fn new(lru: &'a LruArray<N>) -> LruIterator<'a, N> {
	let level = Level::from(LruArray::LEVEL_USER_START + (N - 1));
	let iter  = LevelIterator::new(lru, level);
	LruIterator {
	    lru,
	    level,
	    iter,
	}
    }
}


impl<'a, const N: usize> Iterator for LruIterator<'a, N>
        where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
[(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    type Item = Slot;
    fn next(&mut self) -> Option<Self::Item> {
	while self.level >= LruArray::LEVEL_USER_START {
	    if let Some(slot) = self.iter.next() {
		return Some(slot);
	    }
	    self.level -= 1;
	    self.iter = LevelIterator::new(self.lru, self.level);
	}
	None
    }
}


/// Create shorthand name for a list item
macro_rules! name_index {
    ($name:ident, $list: expr, $ndx:ident) => {
	macro_rules! $name {
	    () => { ($list[$ndx]) }
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
	    // let (a, [second, ..]) = $list.split_at_mut($second);
	    // let (a, [first, ..])  = a.split_at_mut($first);
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


pub(crate) struct LruArray<const N: usize>
where
    [(); 14 - N]: Sized, // 4 bits for levels -> 16 - 2 = 14 levels
    [(); N + 3]: Sized,  // This guards that we have at least one user level
[(); N + 2]: Sized
{
    size:        usize,
    items:       Vec<LruItem>,
    levels:      [LevelState;N + 2],
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

    pub fn new(config: &CacheConfig<N>) -> Self {
	let size = config.blocks;
	//let mut items = Vec::with_capacity(blocks.try_into().unwrap() + N + 2);
	let items = (0 as usize..(size + Self::LEVELS)).map(|i| {
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
	    // let prev = if i == 0 { size + Self::LEVEL_FREE } else { i - 1 };
	    // let next = if i == size - 1 { size + Self::LEVEL_FREE } else { i + 1 };
	    LruItem::new(Slot::from(prev), Slot::from(next), Self::LEVEL_FREE)
	}).collect();

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

	LruArray {
	    size,
	    items,
	    levels,
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

	    name_index!(s_head, self.items, s_head_ndx);
	    name_index!(t_head, self.items, t_head_ndx);

	    while s_level.allocated > s_level.max_size {
		s_level.allocated -= 1;
		t_level.allocated += 1;
		balanced += 1;
		// Take from the tail, give to the head
		let item_ndx: usize  = s_head!().prev().into();  // The index we are moving one level down
		let first_ndx: usize = t_head!().next().into();  // The previously first item on target

		trace!("Moving item {}", item_ndx);

		name_index!(item, self.items, item_ndx);
		name_index!(first, self.items, first_ndx);
		let new_prev = item!().prev();
		// Update heads
		s_head!().set_prev(new_prev);
		t_head!().set_next(item_ndx.into());

		item!().set_next(first_ndx.into());
		item!().set_prev(t_head_ndx.into());
		item!().set_level((ndx - 1).into());

		first!().set_prev(item_ndx.into());
		// NB: Fixup 'source tail' later on when writing to array, as we only need to do it once
		// after all items are balanced

	    }
	    // As noted above, the last item remaining on the source level must have its next ptr fixed
	    let last_ndx: usize = s_head!().prev().into();
	    self.items[last_ndx].set_next(s_head_ndx.into());
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
	let &(mut level_free): &LevelState = &self.levels[usize::from(Self::LEVEL_FREE)];
	if level_free.allocated == 0 {
	    return None;
	}
	level_free.allocated -= 1;

	let user_level    = Self::LEVEL_USER_START + level;
	let free_head_ndx = self.level_head(Self::LEVEL_FREE);
	let free_head     = self.items[free_head_ndx];
	let slot          = free_head.prev();                   // <- Our slot
	let slot_item     = self.items[usize::from(slot)];
	let new_last      = slot_item.prev();

	// Update "new last" (if allocated == 0, this is the head)
	self.items[usize::from(new_last)]
	    .set_next(free_head_ndx.into());

	// Update free head
	self.items[free_head_ndx]
	    .set_prev(new_last);

	// Update user level head
	let level_head_ndx = self.level_head(user_level);
	let level_head = self.items[level_head_ndx];
	let next_slot = level_head.next();
	self.items[usize::from(next_slot)]
	    .set_prev(slot);
	self.items[level_head_ndx]
	    .set_next(slot);
	let slot_item = &mut self.items[usize::from(slot)];
	slot_item.set_next(next_slot);
	slot_item.set_prev(level_head_ndx.into());
	slot_item.set_level(user_level.try_into().unwrap()); // Compile time constraint makes this always work!
	// And update book-keeping
	self.levels[usize::from(user_level)].allocated += 1;
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
	name_index!(t_head, self.items, t_head_ndx);

	// Patch hole P <-> (I) <-> N
	let prev_ndx = usize::from(item!().prev());
	let next_ndx = usize::from(item!().next());
	name_index!(prev, self.items, prev_ndx);
	name_index!(next, self.items, next_ndx);
	prev!().set_next(next_ndx.into());
	next!().set_prev(prev_ndx.into());

	// And then insert item at H <-> I <-> O
	let old_top_ndx = usize::from(t_head!().next());
	name_index!(old_top, self.items, old_top_ndx);
	old_top!().set_prev(slot);
	t_head!().set_next(slot);

	// And update item
	item!().set_prev(t_head_ndx.into());
	item!().set_next(old_top_ndx.into());
	item!().set_level(t_level);
    }

    /// Iterator from high->low
    #[cfg(test)]
    pub fn iter<'a>(&'a self) -> LruIterator<'a, N> {
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use log::{ trace};

    use super::{LruArray, Slot};
    use crate::config::CacheConfig;
    // TODO: Macro for generating tests for different configurations
    const L1: usize = 0;
    const L2: usize = 1;
    const L3: usize = 2;
    const L4: usize = 3;
    const LEVELS: usize = 3;

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

    #[test]
    fn test_lru() {
	let config = CacheConfig::<LEVELS>::default(1000);
	let mut lru = LruArray::new(&config);
	let mut l1_slots = lru.get_slots::<L1, VecDeque<Slot>>(10);
	let mut l2_slots = lru.get_slots::<L2, VecDeque<Slot>>(10);
	let mut l3_slots = lru.get_slots::<L3, VecDeque<Slot>>(10);
	// This must fail at compile time, however how to test it!?
	//let mut l4_slots = lru.get_slots::<L4>(10);

	println!("LEVEL 1 {:?}", l1_slots);
	println!("LEVEL 2 {:?}", l2_slots);
	println!("LEVEL 3 {:?}", l3_slots);

	assert!(compare_level(&l1_slots, lru.level_iter::<L1>()));
	assert!(compare_level(&l2_slots, lru.level_iter::<L2>()));
	assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));

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
	assert!(compare_level(&l1_slots, lru.level_iter::<L1>()));
	assert!(compare_level(&l2_slots, lru.level_iter::<L2>()));
	assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));
	// It should still be in great balance
	let balanced = lru.balance();
	assert!(balanced == 0);
	// But now it will overflow; no problems for now however
	l3_slots.push_back(lru.get(L3).unwrap());
	assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));
	// Balance should put one item to the cleaner
	assert!(lru.balance() == 1);

	// And now should be in balance again
	assert!(lru.balance() == 0);

	// And now all levels should have changed (first item moved down on each level)
	// (i.e. last in comparator list)
	l2_slots.push_back(l3_slots.pop_front().unwrap());
	l1_slots.push_back(l2_slots.pop_front().unwrap());
	l1_slots.pop_front(); // <- And this should be in alloc_free list
	assert!(compare_level(&l2_slots, lru.level_iter::<L2>()));
	assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));

	// Promote the previously "downgraded" items;
	let l2_promote = l1_slots.pop_back().unwrap();
	let l3_promote = l2_slots.pop_back().unwrap();
	l2_slots.push_back(l2_promote);
	l3_slots.push_back(l3_promote);
	lru.promote(l2_promote, Some(1));
	lru.promote(l3_promote, Some(2));

	assert!(compare_level(&l2_slots, lru.level_iter::<L2>()));
	assert!(compare_level(&l3_slots, lru.level_iter::<L3>()));

    }
}
