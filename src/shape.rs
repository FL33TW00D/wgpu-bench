use std::ops::RangeTo;

use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    pub fn new(shape: SmallVec<[usize; 4]>) -> Self {
        Shape(shape)
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = String::from("[");
        for (i, dim) in self.0.iter().enumerate() {
            if i == 0 {
                shape.push_str(&format!("{}", dim));
            } else {
                shape.push_str(&format!("x{}", dim));
            }
        }
        write!(f, "{}]", shape)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::Index<RangeTo<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}
