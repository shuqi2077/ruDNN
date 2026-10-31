use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

pub mod attention;
pub mod moe;
pub mod sol;
pub mod sol_attention;
pub mod sol_gemm;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
pub fn exp_fast(x: f32) -> f32 {
    f32::powf(2.0f32, x * core::f32::consts::LOG2_E)
}

#[ruda]
pub fn block_sum(x: f32, scratch: &mut SharedMemory<f32>, #[comptime] threads: usize) -> f32 {
    let thread = UNIT_POS as usize;
    let mut result = plane_sum(x);
    if comptime!(threads > 32) {
        if thread % 32 == 0 { scratch[thread / 32] = result; }
        sync_ruda();
        let mut part = 0.0f32;
        if thread % 32 < threads / 32 { part = scratch[thread % 32]; }
        result = plane_sum(part);
        sync_ruda();
    }
    result
}

#[ruda(launch)]
pub fn rms<F: Float>(x: &Array<Vector<F, Const<8>>>, residual: &Array<Vector<F, Const<8>>>,
    weight: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<F, Const<8>>>,
    #[comptime] width: usize, #[comptime] threads: usize, #[comptime] rows: usize, #[comptime] add: bool,
    #[define(F)] _dtype: StorageType,
) {
    let local_row = UNIT_POS as usize / threads;
    let row = RUDA_POS_X as usize * rows + local_row;
    let thread = UNIT_POS as usize % threads;
    let valid = row < x.len() / (width / 8);
    let mut cache = Array::<Vector<f32, Const<8>>>::new(width.div_ceil(8 * threads));
    let mut scratch = SharedMemory::<f32>::new(rows * threads / 32);
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..comptime!(width.div_ceil(8 * threads)) {
        let col = thread + j * threads;
        let mut value = Vector::<f32, Const<8>>::new(0.0f32);
        if valid && col < width / 8 {
            value = Vector::cast_from(x[row * (width / 8) + col]);
            if comptime!(add) { value += Vector::<f32, Const<8>>::cast_from(residual[row * (width / 8) + col]); }
        }
        cache[j] = value;
        #[unroll] for d in 0usize..8 { local += value[d] * value[d]; }
    }
    let mut total = plane_sum(local);
    if comptime!(threads > 32) {
        if thread % 32 == 0 { scratch[UNIT_POS as usize / 32] = total; }
        sync_ruda();
        let mut part = 0.0f32;
        if thread % 32 < threads / 32 { part = scratch[local_row * (threads / 32) + thread % 32]; }
        total = plane_sum(part);
    }
    let inv = (total / width as f32 + 1e-6f32).inverse_sqrt();
    #[unroll]
    for j in 0usize..comptime!(width.div_ceil(8 * threads)) {
        let col = thread + j * threads;
        if valid && col < width / 8 {
            let value = cache[j] * Vector::new(inv);
            out[row * (width / 8) + col] = Vector::cast_from(value * Vector::<f32, Const<8>>::cast_from(weight[col]));
        }
    }
}

#[ruda(launch)]
pub fn split_reduce<F: Float>(input: &Array<f32>, output: &mut Array<F>, elements: u32,
    #[comptime] partitions: usize, #[define(F)] _dtype: StorageType,
) {
    let p = ABSOLUTE_POS as usize;
    if p < elements as usize {
        let mut value = input[p];
        #[unroll] for s in 1usize..partitions { value += input[s * elements as usize + p]; }
        output[p] = F::cast_from(value);
    }
}
