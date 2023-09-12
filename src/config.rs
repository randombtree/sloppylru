use std::cmp::max;

pub struct CacheConfig<const N: usize> {
    /// Total block count
    pub(crate) blocks: usize,
    /// The free "slack" that allows for quickly getting a free slot.
    pub(crate) free_slack: usize,
    /// Wake cleaner when the free slack falls below this percentage
    pub(crate) free_hysteresis_pct: u8,
    /// Level fraction (percent) of total cache. Must add up to 100%.
    pub(crate) level_pct: [u8; N],
}


impl<const LEVELS: usize> CacheConfig<LEVELS> {
    /// New level config
    /// free_hysteresis_pct: The proposed wake up for the cleaner, will however always be at least 1 block.
    /// level_pct: The proposed level balance, MUST add up to 100%. Will however be balanced to include slack.
    pub fn new(blocks: usize, free_slack: usize, free_hysteresis_pct: u8, level_pct: [u8; LEVELS])
	       -> CacheConfig<LEVELS> {
	assert!(free_slack < blocks);
	assert!(free_slack > 2); // Allow at least one block for "hysteresis"
	assert!(0 < free_hysteresis_pct && free_hysteresis_pct < 100);
	assert!(level_pct.iter().fold(0, |acc, x| acc + *x as u32) == 100);
	let free_blocks = blocks - free_slack;
	// Now adjust level_pct so that it leaves enough room for free_slack
	let (level_pct, allocated) = {
	    let blocks_f = blocks as f64;
	    let mut target = 0;
	    let mut allocated: usize;
	    let mut level_pct = level_pct.clone();
	    let fold_blocks = |acc: usize, x: &u8| {
		acc + (((*x as f64) / 100.0 * blocks_f).floor() as usize)
	    };
	    loop {
		allocated = level_pct.iter().fold(0, fold_blocks);
		if allocated <= free_blocks { break; }
		level_pct[target] -= 1;
		target = (target + 1) % LEVELS;
	    }
	    (level_pct, allocated)
	};

	// Free slack can grow as the levels are adjusted
	let free_slack = blocks - allocated;
	// Adjust free_hysteresis so that it at least has one block to work with
	let free_hysteresis_pct = {
	    let free_slack = free_slack as f64;
	    let mut free_hysteresis_pct = free_hysteresis_pct;
	    while (((free_hysteresis_pct as f64) / 100.0 * free_slack).floor() as usize) < 1 {
		free_hysteresis_pct += 1;
	    }
	    assert!(free_hysteresis_pct < 100);
	    free_hysteresis_pct
	};
	CacheConfig {
	    blocks,
	    free_slack,
	    free_hysteresis_pct,
	    level_pct,
	}
    }

    /// Make even levels, with the last levels getting padded with the remainder,
    /// e.g. [16, 16, 17, 17, 17, 17] for /6
    pub const fn even_levels() -> [u8; LEVELS] {
	let pct = 100 / LEVELS;
	let start_pad = LEVELS - (100 - (LEVELS * pct));
	let pct = pct as u8;
	// Need to do funny things to keep it const :)
	// Once a const collect into array is implemented in rust, change this..
	let mut levels =  [0; LEVELS];
	let mut ndx = 0;
	while ndx < LEVELS {
	    levels[ndx] = if ndx < start_pad { pct } else { pct + 1 };
	    ndx += 1;
	}
	levels
    }

    /// Construct config for "default" balanced cache, shouldn't be used in production code as the free slack
    /// and hysteresis points can be limiting (TODO: Dynamic slack + hysteresis(TM))
    pub fn default(blocks: usize) -> CacheConfig<LEVELS> {
	let free_slack = max(10, blocks / 100); // Allocate 1% to free machinery
	let level_pct = Self::even_levels();
	let free_hysteresis_pct: u8 = 50;
	Self::new(blocks, free_slack, free_hysteresis_pct, level_pct)
    }

    #[cfg(test)]
    pub(crate) fn level_pct_sum(&self) -> u32 {
	self.level_pct.iter().fold(0, |acc, x| acc + *x as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::CacheConfig;
    #[test]
    fn test_even_levels() {
	assert!(CacheConfig::<1>::even_levels() == [100]);
	assert!(CacheConfig::<2>::even_levels() == [50, 50]);
	assert!(CacheConfig::<3>::even_levels() == [33, 33, 34]);
	assert!(CacheConfig::<4>::even_levels() == [25, 25, 25, 25]);
	assert!(CacheConfig::<6>::even_levels() == [16, 16, 17, 17, 17, 17]);
    }

    #[test]
    fn test_default_config() {
	const BLOCKS: usize = 1000;

	let config = CacheConfig::<1>::default(BLOCKS);
	assert!(config.level_pct.len() == 1);
	assert!(config.level_pct[0]    == 99);
	assert!(config.free_slack      == 10);
	let config = CacheConfig::<3>::default(BLOCKS);
	assert!(config.level_pct.len() == 3);
	assert!(config.level_pct_sum() == 99);
	let config = CacheConfig::<4>::default(BLOCKS);
	assert!(config.level_pct_sum() == 99);
	assert!(config.free_slack      == 10);
	// The equalizer should "steal" the slack pct from the first level
	assert!(config.level_pct[0]    == 24);
	assert!(config.level_pct[3]    == 25);
    }
}
