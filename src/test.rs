use std::sync::{Arc};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{Result as IOResult};
use rand;
use log::{trace};


pub struct TestPath(Arc<TestPathInner>);
#[allow(dead_code)]
struct TestPathInner {
    /// Reference to parent path to hinder removal of parent directory..
    parent: Option<TestPath>,
    path: PathBuf,
}

impl TestPath {
    /// Get a temporary directory, with a descriptive prefix
    /// To the prefix, a random number is added to avoid using
    /// left over paths from crashy tests
    pub fn new(prefix: &str) -> IOResult<TestPath> {
	let r: u32 = rand::random();
	let path = PathBuf::from(format!("{}-{}", prefix, r));
	// If it happens to exist, fail
	fs::create_dir(&path)?;

	Ok(TestPath(
	    Arc::new(TestPathInner {
		parent: Option::None,
		path,
	    })
	))
    }

    /// Get a sub directory with a descriptive suffix
    pub fn subdir(&self, suffix: &str) -> IOResult<TestPath> {
	let r: u32 = rand::random();
	let inner: &TestPathInner = &self.0;
	let path = inner.path.join(format!("{}-{}", suffix, r));
	fs::create_dir(&path)?;
	let parent = Option::Some(TestPath(self.0.clone()));
	Ok(TestPath(
	    Arc::new(TestPathInner {
		parent,
		path,
	    })
	))
    }
}

// Automatic removal of test files
impl Drop for TestPathInner {
    fn drop(&mut self) {
	if self.path.is_dir() {
	    trace!("Dropping test path {:?}", self.path);
	    fs::remove_dir_all(&self.path).unwrap();
	} else {
	    println!("Hmm, {} disappeared?", self.path.to_string_lossy());
	}
    }
}

/// Allow using it as a Path substitute
impl AsRef<Path> for TestPath {
    fn as_ref(&self) -> &Path {
	&self.0.path.as_path()
    }
}

#[cfg(test)]
mod tests {
    use std::path::{PathBuf};
    use crate::test::TestPath;
    /// Test the TestPath implementation
    #[test]
    fn test_test_path() {
	let path = "test_path_test";
	let root_path: Option<PathBuf>;
	{
	    let root = TestPath::new(path).unwrap();
	    root_path = Some(root.as_ref().to_path_buf());
	    assert!(root.as_ref().is_dir(), "Didn't create root path");
	    let sub = root.subdir("subdir1").unwrap();
	    assert!(sub.as_ref().is_dir(), "Didn't create root path");
	    let sub2_path: Option<PathBuf>;
	    {
		let sub2 = root.subdir("subdir2").unwrap();
		sub2_path = Some(sub2.as_ref().to_path_buf());
		print!("{}", sub2.as_ref().to_string_lossy());
		assert!(sub2.as_ref().is_dir(), "Didn't create root path");
	    }
	    assert!(!sub2_path.unwrap().is_dir(), "Didn't properly remove subpath");
	}
	assert!(!root_path.unwrap().is_dir(), "Didn't properly remove subpath");
    }
}

