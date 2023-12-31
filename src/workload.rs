#[derive(Debug, derive_new::new)]
pub struct WorkgroupCount(pub u32, pub u32, pub u32); //Analagous to gridDim in CUDA

impl WorkgroupCount {
    pub fn as_tuple(&self) -> (u32, u32, u32) {
        (self.0, self.1, self.2)
    }
}

#[macro_export]
macro_rules! wgc {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::WorkgroupCount::new($x, $y, $z)
    };
}

#[derive(Debug, derive_new::new)]
pub struct WorkgroupSize(pub u32, pub u32, pub u32); //Analagous to blockDim in CUDA

impl WorkgroupSize {
    pub fn total(&self) -> u32 {
        self.0 * self.1 * self.2
    }
}

#[macro_export]
macro_rules! wgs {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::WorkgroupSize::new($x, $y, $z)
    };
}

///The Workload represents the entire piece of work.
///For more read: https://surma.dev/things/webgpu/
#[derive(Debug)]
pub struct Workload {
    size: WorkgroupSize,
    count: WorkgroupCount,
}

impl Workload {
    pub fn new(size: WorkgroupSize, count: WorkgroupCount) -> Self {
        Self { size, count }
    }

    pub fn count(&self) -> &WorkgroupCount {
        &self.count
    }

    pub fn size(&self) -> &WorkgroupSize {
        &self.size
    }
}

///Used to determine which limit applies
#[derive(Debug, Clone)]
pub enum WorkloadDim {
    X,
    Y,
    Z,
}

impl Workload {
    pub const MAX_WORKGROUP_SIZE_X: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Y: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Z: usize = 64;
    pub const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: usize = 65535;

    pub fn ceil(num: usize, div: usize) -> usize {
        (num + div - 1) / div
    }
}
