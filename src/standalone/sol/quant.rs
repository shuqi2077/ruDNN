//! FP8 (E4M3) blockwise fake-quantization for the Quant suite: reproduce the
//! reference quantize -> dequantize numerics (scale = max(amax / 448, 1e-12),
//! q = e4m3(clamp(x / scale, +-448)), x' = f32(q) * scale) and emit O (bf16) for
//! the tcgen05 GEMMs. Activations use 1x128 blocks, weights 128x128 blocks.
use ruda_core::e4m3;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda]
fn fake_quant(x: f32, scale: f32) -> f32 {
    let mut v = x / scale;
    v = f32::max(f32::min(v, 448.0f32), -448.0f32);
    f32::cast_from(e4m3::cast_from(v)) * scale
}

/// Activation 1x128 blocks: one warp per (row, block), lane owns 4 elements.
/// `given`: scales (rows, cols / 128) are inputs, else computed (and written to scales).
#[ruda(launch)]
pub fn fq_act<F: Float, O: Float>(x: &Array<F>, scales: &mut Array<f32>, out: &mut Array<O>, rows: u32, cols: u32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType, #[comptime] given: bool,
) {
    let blocks = cols as usize / 128;
    let item = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if item < rows as usize * blocks {
        let row = item / blocks;
        let block = item % blocks;
        let base = row * cols as usize + block * 128;
        let mut v = Array::<f32>::new(4usize);
        let mut amax = 0.0f32;
        #[unroll]
        for j in 0usize..4 {
            v[j] = f32::cast_from(x[base + j * 32 + lane]);
            amax = f32::max(amax, f32::abs(v[j]));
        }
        let mut scale = 0.0f32;
        if comptime!(given) {
            scale = scales[item];
        } else {
            scale = f32::max(plane_max(amax) / 448.0f32, 1e-12f32);
            if lane == 0 { scales[item] = scale; }
        }
        #[unroll]
        for j in 0usize..4 { out[base + j * 32 + lane] = O::cast_from(fake_quant(v[j], scale)); }
    }
}

/// Weight 128x128 blocks of W (N, K): one 256-thread block per tile, grid (K / 128, N / 128).
/// Scale of tile (nb, kb) lives at nb * s_n + kb * s_k (given or computed).
#[ruda(launch)]
pub fn fq_weight<F: Float, O: Float>(w: &Array<Vector<F, Const<8>>>, scales: &mut Array<f32>, out: &mut Array<Vector<O, Const<8>>>, cols: u32,
    s_n: u32, s_k: u32, #[define(F)] _f: StorageType, #[define(O)] _o: StorageType, #[comptime] given: bool,
) {
    let kb = RUDA_POS_X as usize;
    let nb = RUDA_POS_Y as usize;
    let t = UNIT_POS as usize;
    let mut cache = Array::<Vector<f32, Const<8>>>::new(8usize);
    let mut amax = 0.0f32;
    #[unroll]
    for j in 0usize..8 {
        let e = t + 256 * j;
        let index = (nb * 128 + e / 16) * (cols as usize / 8) + kb * 16 + e % 16;
        let mut value = Vector::<f32, Const<8>>::new(0.0f32);
        if index < w.len() { value = Vector::<f32, Const<8>>::cast_from(w[index]); }
        cache[j] = value;
        if comptime!(!given) {
            #[unroll]
            for d in 0usize..8 { amax = f32::max(amax, f32::abs(value[d])); }
        }
    }
    let mut shared = SharedMemory::<f32>::new(8usize);
    let slot = nb * s_n as usize + kb * s_k as usize;
    let mut scale = 0.0f32;
    if comptime!(given) { scale = scales[slot]; } else {
        let warp_max = plane_max(amax);
        if t % 32 == 0 { shared[t / 32] = warp_max; }
        sync_ruda();
        let mut part = 0.0f32;
        if t % 32 < 8 { part = shared[t % 32]; }
        let total = plane_max(part);
        scale = f32::max(total / 448.0f32, 1e-12f32);
        if t == 0 { scales[slot] = scale; }
    }
    #[unroll]
    for j in 0usize..8 {
        let e = t + 256 * j;
        let index = (nb * 128 + e / 16) * (cols as usize / 8) + kb * 16 + e % 16;
        if index < out.len() {
            let value = cache[j];
            let mut result = Vector::<O, Const<8>>::empty();
            #[unroll]
            for d in 0usize..8 { result[d] = O::cast_from(fake_quant(value[d], scale)); }
            out[index] = result;
        }
    }
}

/// Dequantize E4M3 (u8 storage) with block scales: element (r, c) uses
/// scales[(r / block_rows) * s_r + (c / 128) * s_c].
#[ruda(launch)]
pub fn dq_fp8<O: Float>(x: &Array<u8>, scales: &Array<f32>, out: &mut Array<O>, count: u32, cols: u32,
    block_rows: u32, s_r: u32, s_c: u32, #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let r = i / cols as usize;
        let c = i % cols as usize;
        let s = scales[(r / block_rows as usize) * s_r as usize + (c / 128) * s_c as usize];
        out[i] = O::cast_from(f32::cast_from(e4m3::reinterpret(x[i])) * s);
    }
}

/// Quant/014: YaRN cos/sin with FP8 fake-quantized frequencies (one warp per
/// position, dim 64 = 2 x 32 lanes): f = t * inv[lane], 1x128 block scale
/// (zero padded), cos / sin of the dequantized f times mscale, as O.
#[ruda(launch)]
pub fn yarn_cos_sin<O: Float>(positions: &Array<i64>, inv_freq: &Array<f32>, cos: &mut Array<O>, sin: &mut Array<O>,
    rows: u32, mscale: f32, #[define(O)] _o: StorageType,
) {
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if row < rows as usize {
        let f = f32::cast_from(positions[row]) * inv_freq[lane];
        let scale = f32::max(plane_max(f32::abs(f)) / 448.0f32, 1e-12f32);
        let d = fake_quant(f, scale);
        let c = O::cast_from(d.cos() * mscale);
        let s = O::cast_from(d.sin() * mscale);
        cos[row * 64 + lane] = c;
        cos[row * 64 + 32 + lane] = c;
        sin[row * 64 + lane] = s;
        sin[row * 64 + 32 + lane] = s;
    }
}

/// Quant/023: dt = clamp(softplus(f32(dt_proj)), lo, hi); dA = exp(dt * -exp(A_log));
/// dB[b, l, h, n] = dt * B[b, l, h / (H / G), n]. One thread per (b, l, h, n).
#[ruda(launch)]
pub fn ssm_discretize<F: Float>(dt_proj: &Array<F>, a_log: &Array<f32>, bmat: &Array<F>,
    dt_out: &mut Array<F>, da: &mut Array<f32>, db: &mut Array<f32>, count: u32, heads: u32, groups: u32, state: u32,
    lo: f32, hi: f32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let n = i % state as usize;
        let h = (i / state as usize) % heads as usize;
        let bl = i / (state as usize * heads as usize);
        let x = f32::cast_from(dt_proj[bl * heads as usize + h]);
        let mut sp = x;
        if x <= 20.0f32 { sp = (1.0f32 + x.exp()).ln(); }
        let dt = f32::min(f32::max(sp, lo), hi);
        let g = h / (heads as usize / groups as usize);
        db[i] = dt * f32::cast_from(bmat[(bl * groups as usize + g) * state as usize + n]);
        if n == 0 {
            dt_out[bl * heads as usize + h] = F::cast_from(dt);
            da[bl * heads as usize + h] = (dt * -(a_log[h].exp())).exp();
        }
    }
}
