pub trait DataType:
    Clone
    + std::fmt::Debug
    + std::fmt::Display
    + PartialEq
    + 'static
    + num_traits::Zero
    + Send
    + Sync
    + bytemuck::Pod
{
    fn dt() -> DType;

    fn one() -> Self;
}

macro_rules! map_type {
    ($t:ty, $v:ident) => {
        impl DataType for $t {
            fn dt() -> DType {
                DType::$v
            }

            fn one() -> Self {
                1 as Self
            }
        }
    };
}

map_type!(f32, F32);
map_type!(i32, I32);
map_type!(u32, U32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
pub enum DType {
    #[default]
    F32,
    I32,
    U32,
}

impl DType {
    pub fn size_of(self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::U32 => std::mem::size_of::<u32>(),
            _ => unimplemented!(),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::I32 => write!(f, "i32"),
            DType::U32 => write!(f, "u32"),
            _ => unimplemented!(),
        }
    }
}

fn handle_type_str(ts: npyz::TypeStr) -> DType {
    match ts.endianness() {
        npyz::Endianness::Little => match (ts.type_char(), ts.size_field()) {
            (npyz::TypeChar::Float, 4) => DType::F32,
            (npyz::TypeChar::Int, 4) => DType::I32,
            (npyz::TypeChar::Uint, 4) => DType::U32,
            (t, s) => unimplemented!("{} {}", t, s),
        },
        _ => unimplemented!(),
    }
}

impl From<npyz::DType> for DType {
    fn from(dtype: npyz::DType) -> Self {
        match dtype {
            npyz::DType::Plain(ts) => handle_type_str(ts),
            _ => unimplemented!(),
        }
    }
}
