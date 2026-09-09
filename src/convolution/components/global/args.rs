use ruda_kernel::dsl as cubecl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

use crate::convolution::components::ConvolutionOperation;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod<u32>,
    #[cube(comptime)]
    pub operation: ConvolutionOperation,
}
