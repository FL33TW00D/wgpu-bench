use encase::{private::WriteInto, ShaderType};
pub const UNIFORM_ALIGN: usize = 256;

pub trait OpMetadata: Sized + ShaderType + WriteInto {
    const __IS_VALID_META: () = {
        assert!(std::mem::size_of::<Self>() <= UNIFORM_ALIGN);
    };

    fn n_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}
