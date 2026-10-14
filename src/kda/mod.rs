//! Channel-gated delta attention, following the Kimi/FLA KDA equations.
//! See LICENSE.fla for the pinned algorithm reference and upstream license.
mod chunk;
mod kernel;
mod prepared;

pub use chunk::{chunk_kda, chunk_kda_backward, chunk_kda_with_history};
pub use prepared::{prepare_chunk_kda, prepare_chunk_kda_backward};

use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use std::fmt;

/// Q/K [B,H,T,K], V [B,H,T,V]; Q/K are already normalized if required.
/// Beta [B,H,T], log_decay [B,H,T,K] and state [B,H,K,V] use FP32.
/// Grouped query heads are expanded by the caller, not folded into decay heads.
pub struct KdaInput<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
    pub value: RudaTensor<R>,
    pub beta: RudaTensor<R>,
    pub log_decay: RudaTensor<R>,
    pub initial_state: RudaTensor<R>,
    pub query_scale: f32,
}

pub struct KdaOutput<R: Runtime> {
    pub output: RudaTensor<R>,
    pub final_state: RudaTensor<R>,
    /// Pre-chunk states [ceil(T/C),B,H,K,V], or None for inference.
    pub boundaries: Option<RudaTensor<R>>,
}

pub struct KdaHistoryOutput<R: Runtime> {
    pub result: KdaOutput<R>,
    pub states: Vec<RudaTensor<R>>,
}

pub struct KdaGradients<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
    pub value: RudaTensor<R>,
    pub beta: RudaTensor<R>,
    pub log_decay: RudaTensor<R>,
    pub initial_state: RudaTensor<R>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaError(pub &'static str);

impl fmt::Display for KdaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0) }
}
impl std::error::Error for KdaError {}

fn check_size(shape: &[usize]) -> Result<(), KdaError> {
    if shape.iter().any(|&d| d > u32::MAX as usize)
        || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .is_none_or(|n| n > u32::MAX as usize)
    { return Err(KdaError("KDA tensor/workspace exceeds U32 indexing")); }
    Ok(())
}

fn check_tensor<R: Runtime>(tensor: &RudaTensor<R>, reference: &RudaTensor<R>,
    shape: &[usize], dtype: DType) -> Result<(), KdaError>
{
    check_size(shape)?;
    if &tensor.meta.shape()[..] != shape || tensor.dtype != dtype || tensor.qparams.is_some() {
        return Err(KdaError("incompatible KDA tensor shape or dtype"));
    }
    if tensor.device.to_id() != reference.device.to_id() {
        return Err(KdaError("KDA tensors must share a device"));
    }
    Ok(())
}

fn validate<R: Runtime>(input: &KdaInput<R>, chunk: usize) -> Result<[usize; 5], KdaError> {
    let q = &input.query;
    if q.meta.num_dims() != 4 || input.value.meta.num_dims() != 4
        || !matches!(q.dtype, DType::F32 | DType::F16 | DType::BF16)
    { return Err(KdaError("KDA requires four-dimensional FP32/FP16/BF16 Q/K/V")); }
    let [b, h, t, k]: [usize; 4] = q.meta.shape()[..].try_into().unwrap();
    let v = input.value.meta.shape()[3];
    if h == 0 || k == 0 || v == 0 || chunk == 0 || chunk > u32::MAX as usize
        || !input.query_scale.is_finite()
    { return Err(KdaError("KDA dimensions/chunk must be positive and scale finite")); }
    for (tensor, shape, dtype) in [
        (q, vec![b,h,t,k], q.dtype), (&input.key, vec![b,h,t,k], q.dtype),
        (&input.value, vec![b,h,t,v], q.dtype), (&input.beta, vec![b,h,t], DType::F32),
        (&input.log_decay, vec![b,h,t,k], DType::F32),
        (&input.initial_state, vec![b,h,k,v], DType::F32),
    ] { check_tensor(tensor, q, &shape, dtype)?; }
    let c = chunk.min(t.max(1));
    for width in [c,k,v] { check_size(&[b,h,c,width])?; }
    let shared = c.checked_mul(4 * 4);
    if shared.is_none_or(|n| n > q.client.properties().hardware.max_shared_memory_size as usize) {
        return Err(KdaError("KDA triangular solve exceeds shared memory"));
    }
    Ok([b,h,t,k,v])
}
