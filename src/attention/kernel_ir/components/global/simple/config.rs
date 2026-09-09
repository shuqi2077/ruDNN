use ruda_kernel::dsl as cubecl;
use ruda_kernel::dsl::CubeDim;
use rublas::kernel_ir::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, memory::GlobalMemoryConfig,
};

use crate::attention::kernel_ir::components::{global::GlobalAttentionConfig, stage::StageAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SimpleGlobalAttentionConfig<S: StageAttentionConfig> {
    pub stage_config: S,
    pub key_reader_config: GlobalReaderConfig,
    pub value_reader_config: GlobalReaderConfig,
    pub query_gmem_config: GlobalMemoryConfig,
    pub mask_gmem_config: GlobalMemoryConfig,
    pub writer_config: GlobalWriterConfig,
}

impl<S: StageAttentionConfig> GlobalAttentionConfig for SimpleGlobalAttentionConfig<S> {
    type StageConfig = S;

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(
            self.stage_config.plane_dim(),
            self.stage_config.num_planes(),
        )
    }

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }
}
