use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(SmallVec<[usize; 4]>);

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
