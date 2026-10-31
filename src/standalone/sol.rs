//! Fused SOL-ExecBench kernels (L1 memory-bound group), tuned for B200:
//! one pass over HBM per problem, 16-byte vector accesses where layouts allow,
//! row reductions kept in registers, reference rounding points reproduced.
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

use super::block_sum;

pub mod common;
pub mod l1c;
pub mod l1d;
pub mod moe;
pub mod l2a;
pub mod quant;
pub mod nvfp4;
pub mod vision;
pub mod spectral;
pub mod spatial;
pub mod scan_blocks;
pub mod routing_ext;
pub mod audio_relative;
pub mod multimodal_positions;

// ------------------------------------------------------------------ helpers

/// Round an f32 through storage type `F` (reproduces an intermediate cast).
#[ruda]
fn round_to<F: Float>(value: f32) -> f32 {
    f32::cast_from(F::cast_from(value))
}

// ------------------------------------------------------------------ L1/011

/// cos/sin table (rows, 128, 2) for `positions[row] * inv_freq[d % 64]`.
/// One thread writes four (cos, sin) pairs to both halves of the head dim.
#[ruda(launch)]
pub fn rope_table<F: Float>(positions: &Array<i64>, inv_freq: &Array<f32>,
    out: &mut Array<Vector<F, Const<8>>>, scale: f32, #[define(F)] _dtype: StorageType,
) {
    let task = ABSOLUTE_POS as usize;
    if task < positions.len() * 16 {
        let row = task / 16;
        let group = task % 16;
        let position = f32::cast_from(positions[row]);
        let mut pairs = Vector::<F, Const<8>>::empty();
        #[unroll]
        for i in 0usize..4 {
            let angle = inv_freq[group * 4 + i] * position;
            pairs[2 * i] = F::cast_from(angle.cos() * scale);
            pairs[2 * i + 1] = F::cast_from(angle.sin() * scale);
        }
        out[row * 32 + group] = pairs;
        out[row * 32 + 16 + group] = pairs;
    }
}

// ------------------------------------------------------------------ L1/012

/// cos/sin of cat(freqs, freqs) (rows, 128) from freqs (rows, 64), bf16 out.
#[ruda(launch)]
pub fn cos_sin_halves<F: Float>(freqs: &Array<Vector<f32, Const<4>>>,
    cos_out: &mut Array<Vector<F, Const<4>>>, sin_out: &mut Array<Vector<F, Const<4>>>,
    scale: f32, #[define(F)] _dtype: StorageType,
) {
    let task = ABSOLUTE_POS as usize;
    if task < freqs.len() {
        let row = task / 16;
        let group = task % 16;
        let f = freqs[task];
        let mut c = Vector::<F, Const<4>>::empty();
        let mut s = Vector::<F, Const<4>>::empty();
        #[unroll]
        for i in 0usize..4 {
            c[i] = F::cast_from(f[i].cos() * scale);
            s[i] = F::cast_from(f[i].sin() * scale);
        }
        cos_out[row * 32 + group] = c;
        cos_out[row * 32 + 16 + group] = c;
        sin_out[row * 32 + group] = s;
        sin_out[row * 32 + 16 + group] = s;
    }
}

// ------------------------------------------------------------------ L1/013

/// Residual RMSNorm backward rows (hidden = 8 * threads). Writes grad_x twice
/// (bf16) and per-block column partials of sum(grad * normalized).
#[ruda(launch)]
pub fn rms_backward_rows<F: Float>(grad: &Array<Vector<F, Const<8>>>, normalized: &Array<Vector<f32, Const<4>>>,
    rstd: &Array<f32>, weight: &Array<Vector<f32, Const<4>>>,
    grad_x: &mut Array<Vector<F, Const<8>>>, grad_residual: &mut Array<Vector<F, Const<8>>>,
    partial: &mut Array<Vector<f32, Const<4>>>,
    #[comptime] threads: usize, #[define(F)] _dtype: StorageType,
) {
    let thread = UNIT_POS as usize;
    let width = comptime!(threads * 8);
    let mut scratch = SharedMemory::<f32>::new(comptime!(threads / 32));
    let w0 = weight[thread * 2];
    let w1 = weight[thread * 2 + 1];
    let mut acc0 = Vector::<f32, Const<4>>::new(0.0f32);
    let mut acc1 = Vector::<f32, Const<4>>::new(0.0f32);
    let rows = rstd.len();
    let mut row = RUDA_POS_X as usize;
    while row < rows {
        let g = Vector::<f32, Const<8>>::cast_from(grad[row * threads + thread]);
        let n0 = normalized[(row * threads + thread) * 2];
        let n1 = normalized[(row * threads + thread) * 2 + 1];
        let mut local = 0.0f32;
        let mut gn = Vector::<f32, Const<8>>::empty();
        #[unroll]
        for i in 0usize..4 {
            gn[i] = g[i] * w0[i];
            gn[i + 4] = g[i + 4] * w1[i];
            local += gn[i] * n0[i] + gn[i + 4] * n1[i];
            acc0[i] += g[i] * n0[i];
            acc1[i] += g[i + 4] * n1[i];
        }
        let mean = block_sum(local, &mut scratch, threads) / width as f32;
        let r = rstd[row];
        let mut out = Vector::<F, Const<8>>::empty();
        #[unroll]
        for i in 0usize..4 {
            out[i] = F::cast_from(r * (gn[i] - mean * n0[i]));
            out[i + 4] = F::cast_from(r * (gn[i + 4] - mean * n1[i]));
        }
        grad_x[row * threads + thread] = out;
        grad_residual[row * threads + thread] = out;
        row += RUDA_COUNT_X as usize;
    }
    let block = RUDA_POS_X as usize;
    partial[(block * threads + thread) * 2] = acc0;
    partial[(block * threads + thread) * 2 + 1] = acc1;
}

/// Sum per-block column partials (blocks, width) into out (width).
#[ruda(launch)]
pub fn column_sum(partial: &Array<f32>, out: &mut Array<f32>) {
    let column = ABSOLUTE_POS as usize;
    let width = out.len();
    if column < width {
        let blocks = partial.len() / width;
        let mut total = 0.0f32;
        for block in 0usize..blocks {
            total += partial[block * width + column];
        }
        out[column] = total;
    }
}

// ------------------------------------------------------------------ L1/014

/// d(position) of a scaled RoPE table: one warp per (batch, seq) row.
#[ruda(launch)]
pub fn rope_position_grad<F: Float>(grad_cos: &Array<F>, grad_sin: &Array<F>, emb: &Array<f32>,
    inv_freq: &Array<f32>, out: &mut Array<f32>, seq: u32, scale: f32, #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS_PLANE as usize;
    if row < out.len() {
        let batch = row / seq as usize;
        let mut local = 0.0f32;
        #[unroll]
        for j in 0usize..2 {
            let d = lane + j * 32;
            let base = row * 128;
            let gc = f32::cast_from(grad_cos[base + d]) + f32::cast_from(grad_cos[base + d + 64]);
            let gs = f32::cast_from(grad_sin[base + d]) + f32::cast_from(grad_sin[base + d + 64]);
            let e = emb[base + d];
            let grad = gc * (-e.sin()) * scale + gs * e.cos() * scale;
            local += inv_freq[batch * 64 + d] * grad;
        }
        let total = plane_sum(local);
        if lane == 0 { out[row] = total; }
    }
}

// ------------------------------------------------------------------ L1/016

/// inv_freq[i] = 1 / theta^(2i / 128).
#[ruda(launch)]
pub fn rope_inverse_frequency(out: &mut Array<f32>, theta: f32) {
    let i = UNIT_POS as usize;
    if i < out.len() {
        let exponent = f32::cast_from(i * 2) / 128.0f32;
        out[i] = 1.0f32 / theta.powf(exponent);
    }
}

// ------------------------------------------------------------------ L1/019

/// Vision RoPE backward for (seq, 16 heads, 128): one block of 64 threads per
/// sequence position, thread `d` owns the dimension pair (d, d + 64).
#[ruda(launch)]
pub fn rope_vision_backward(grad_q_embed: &Array<f32>, grad_k_embed: &Array<f32>, q: &Array<f32>, k: &Array<f32>,
    embeddings: &Array<f32>, grad_q: &mut Array<f32>, grad_k: &mut Array<f32>, grad_embeddings: &mut Array<f32>,
) {
    let s = RUDA_POS_X as usize;
    let d = UNIT_POS as usize;
    let e0 = embeddings[s * 128 + d];
    let e1 = embeddings[s * 128 + d + 64];
    let c0 = e0.cos();
    let s0 = e0.sin();
    let c1 = e1.cos();
    let s1 = e1.sin();
    let mut cos0 = 0.0f32;
    let mut cos1 = 0.0f32;
    let mut sin0 = 0.0f32;
    let mut sin1 = 0.0f32;
    #[unroll]
    for h in 0usize..16 {
        let base = (s * 16 + h) * 128;
        let gq0 = grad_q_embed[base + d];
        let gq1 = grad_q_embed[base + d + 64];
        let gk0 = grad_k_embed[base + d];
        let gk1 = grad_k_embed[base + d + 64];
        let q0 = q[base + d];
        let q1 = q[base + d + 64];
        let k0 = k[base + d];
        let k1 = k[base + d + 64];
        grad_q[base + d] = gq0 * c0 + gq1 * s1;
        grad_q[base + d + 64] = gq1 * c1 - gq0 * s0;
        grad_k[base + d] = gk0 * c0 + gk1 * s1;
        grad_k[base + d + 64] = gk1 * c1 - gk0 * s0;
        cos0 += gq0 * q0 + gk0 * k0;
        cos1 += gq1 * q1 + gk1 * k1;
        sin0 += gq0 * (-q1) + gk0 * (-k1);
        sin1 += gq1 * q0 + gk1 * k0;
    }
    grad_embeddings[s * 128 + d] = cos0 * (-s0) + sin0 * c0;
    grad_embeddings[s * 128 + d + 64] = cos1 * (-s1) + sin1 * c1;
}

// ------------------------------------------------------------------ L1/024

/// Partial d(inv_freq) for 2D vision RoPE (head_dim 72, quarter 18):
/// 16 patches x 18 frequencies per block step, block partials (blocks, 18).
#[ruda(launch)]
pub fn rope_grid_frequency_partial(grad_cos: &Array<f32>, grad_sin: &Array<f32>, positions: &Array<i64>,
    emb: &Array<f32>, partial: &mut Array<f32>,
) {
    let thread = UNIT_POS as usize;
    let j = thread % 18;
    let lane = thread / 18;
    let patches = positions.len() / 2;
    let mut shared = SharedMemory::<f32>::new(288usize);
    let mut total = 0.0f32;
    let mut patch = RUDA_POS_X as usize * 16 + lane;
    while patch < patches {
        let base = patch * 72;
        let mut axis = Array::<f32>::new(2usize);
        #[unroll]
        for a in 0usize..2 {
            let i = a * 18 + j;
            let lo = -grad_cos[base + i] * emb[base + i].sin() + grad_sin[base + i] * emb[base + i].cos();
            let hi = -grad_cos[base + i + 36] * emb[base + i + 36].sin() + grad_sin[base + i + 36] * emb[base + i + 36].cos();
            axis[a] = lo + hi;
        }
        total += f32::cast_from(positions[patch * 2]) * axis[0] + f32::cast_from(positions[patch * 2 + 1]) * axis[1];
        patch += RUDA_COUNT_X as usize * 16;
    }
    shared[thread] = total;
    sync_ruda();
    if lane == 0 {
        let mut sum = 0.0f32;
        #[unroll]
        for l in 0usize..16 { sum += shared[l * 18 + j]; }
        partial[RUDA_POS_X as usize * 18 + j] = sum;
    }
}

// ------------------------------------------------------------------ L1/025

/// tanh-approximated GELU in fp32 with the reference operation order.
#[ruda(launch)]
pub fn gelu_tanh_f32(x: &Array<Vector<f32, Const<4>>>, out: &mut Array<Vector<f32, Const<4>>>) {
    let p = ABSOLUTE_POS as usize;
    if p < x.len() {
        let v = x[p];
        let mut result = Vector::<f32, Const<4>>::empty();
        #[unroll]
        for i in 0usize..4 {
            let value = v[i];
            let cubed = value * value * value;
            let inner = 0.7978845608028654f32 * (value + 0.044715f32 * cubed);
            result[i] = 0.5f32 * value * (1.0f32 + inner.tanh());
        }
        out[p] = result;
    }
}

// ------------------------------------------------------------------ L1/028

/// Hybrid attention masks: full = (col > row + past) per (row, col) of each
/// (target, source) plane, sliding = all false. 16 bools per thread step.
#[ruda(launch)]
pub fn hybrid_masks(full: &mut Array<Vector<u32, Const<4>>>, sliding: &mut Array<Vector<u32, Const<4>>>,
    target: u32, source: u32, past: u32,
) {
    let plane = target * source;
    let vectors = u32::cast_from(full.len());
    let mut v = u32::cast_from(ABSOLUTE_POS);
    while v < vectors {
        let start = ((v % plane) * 16u32) % plane;
        let mut row = start / source;
        let mut col = start % source;
        let mut words = Vector::<u32, Const<4>>::new(0u32);
        #[unroll]
        for w in 0usize..4 {
            let mut word = 0u32;
            #[unroll]
            for b in 0u32..4 {
                if col > row + past { word |= 1u32 << (b * 8u32); }
                col += 1u32;
                if col == source {
                    col = 0u32;
                    row += 1u32;
                    if row == target { row = 0u32; }
                }
            }
            words[w] = word;
        }
        full[v as usize] = words;
        sliding[v as usize] = Vector::<u32, Const<4>>::new(0u32);
        v += RUDA_COUNT_X * RUDA_DIM_X;
    }
}

// ------------------------------------------------------------------ RMSNorm rows (L1/033, L1/069)

/// Row RMSNorm with 16-byte vectors: `pre_add` rounds x + residual to F and the
/// normalized value to F before the weight multiply (L1/069); `post_add` adds
/// the residual after rounding the weighted value (L1/033).
#[ruda(launch)]
pub fn rms_rows<F: Float>(x: &Array<Vector<F, Const<8>>>, residual: &Array<Vector<F, Const<8>>>,
    weight: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<F, Const<8>>>, eps: f32,
    #[comptime] width: usize, #[comptime] threads: usize, #[comptime] pre_add: bool, #[comptime] post_add: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let per = comptime!(width / (8 * threads));
    let mut cache = Array::<Vector<f32, Const<8>>>::new(per);
    let mut scratch = SharedMemory::<f32>::new(comptime!(threads / 32));
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..per {
        let col = thread + j * threads;
        let mut value = Vector::<f32, Const<8>>::cast_from(x[row * (width / 8) + col]);
        if comptime!(pre_add) {
            value += Vector::<f32, Const<8>>::cast_from(residual[row * (width / 8) + col]);
            value = Vector::<f32, Const<8>>::cast_from(Vector::<F, Const<8>>::cast_from(value));
        }
        cache[j] = value;
        #[unroll]
        for d in 0usize..8 { local += value[d] * value[d]; }
    }
    let inv = (block_sum(local, &mut scratch, threads) / width as f32 + eps).inverse_sqrt();
    #[unroll]
    for j in 0usize..per {
        let col = thread + j * threads;
        let mut value = cache[j] * Vector::new(inv);
        if comptime!(pre_add) {
            value = Vector::<f32, Const<8>>::cast_from(Vector::<F, Const<8>>::cast_from(value));
        }
        let scaled = value * Vector::<f32, Const<8>>::cast_from(weight[col]);
        if comptime!(post_add) {
            let rounded = Vector::<f32, Const<8>>::cast_from(Vector::<F, Const<8>>::cast_from(scaled));
            out[row * (width / 8) + col] = Vector::cast_from(rounded + Vector::<f32, Const<8>>::cast_from(residual[row * (width / 8) + col]));
        } else {
            out[row * (width / 8) + col] = Vector::cast_from(scaled);
        }
    }
}

// ------------------------------------------------------------------ L1/038

/// Per-head RMSNorm of q and k (head_dim 128, fp32): one warp per row,
/// RUDA_POS_Y selects q (0) or k (1).
#[ruda(launch)]
pub fn head_rms_qk(query: &Array<Vector<f32, Const<4>>>, key: &Array<Vector<f32, Const<4>>>,
    weight_q: &Array<Vector<f32, Const<4>>>, weight_k: &Array<Vector<f32, Const<4>>>,
    query_out: &mut Array<Vector<f32, Const<4>>>, key_out: &mut Array<Vector<f32, Const<4>>>,
    eps: f32, #[comptime] heads: usize,
) {
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS_PLANE as usize;
    let rows = query.len() / 32;
    if row < rows {
        let head = row % heads;
        let index = row * 32 + lane;
        let mut value = query[index];
        let mut w = weight_q[head * 32 + lane];
        if RUDA_POS_Y == 1u32 {
            value = key[index];
            w = weight_k[head * 32 + lane];
        }
        let mut local = 0.0f32;
        #[unroll]
        for i in 0usize..4 { local += value[i] * value[i]; }
        let inv = (plane_sum(local) / 128.0f32 + eps).inverse_sqrt();
        let result = value * Vector::new(inv) * w;
        if RUDA_POS_Y == 1u32 { key_out[index] = result; } else { query_out[index] = result; }
    }
}

// ------------------------------------------------------------------ L1/039

/// Instance-norm backward pass 1: per (plane, chunk) partial sums of
/// go * (x - mean), go and (x - mean). Grid (chunks, planes), 256 threads.
#[ruda(launch)]
pub fn instance_norm_partial(grad_output: &Array<f32>, x: &Array<f32>, mean: &Array<f32>,
    partial: &mut Array<f32>, spatial: u32, chunk: u32,
) {
    let plane = RUDA_POS_Y as usize;
    let thread = UNIT_POS as usize;
    let m = mean[plane];
    let start = RUDA_POS_X as usize * chunk as usize;
    let mut end = start + chunk as usize;
    if end > spatial as usize { end = spatial as usize; }
    let base = plane * spatial as usize;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = start + thread;
    while i < end {
        let g = grad_output[base + i];
        let c = x[base + i] - m;
        s1 += g * c;
        s2 += g;
        s3 += c;
        i += 256;
    }
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let t1 = block_sum(s1, &mut scratch, 256usize);
    let t2 = block_sum(s2, &mut scratch, 256usize);
    let t3 = block_sum(s3, &mut scratch, 256usize);
    if thread == 0 {
        let slot = (plane * RUDA_COUNT_X as usize + RUDA_POS_X as usize) * 3;
        partial[slot] = t1;
        partial[slot + 1] = t2;
        partial[slot + 2] = t3;
    }
}

/// Instance-norm backward pass 2: one block per channel, thread per sample.
/// Produces per-plane coefficients (a, b, c) with grad_input = go*a + xc*b + c,
/// plus grad_weight / grad_bias per channel.
#[ruda(launch)]
pub fn instance_norm_finalize(partial: &Array<f32>, weight: &Array<f32>, std: &Array<f32>,
    coefficients: &mut Array<f32>, grad_weight: &mut Array<f32>, grad_bias: &mut Array<f32>,
    samples: u32, chunks: u32, spatial: u32,
) {
    let channel = RUDA_POS_X as usize;
    let channels = weight.len();
    let n = UNIT_POS as usize;
    let mut gw = 0.0f32;
    let mut gb = 0.0f32;
    if n < samples as usize {
        let plane = n * channels + channel;
        let mut p1 = 0.0f32;
        let mut p2 = 0.0f32;
        let mut p3 = 0.0f32;
        for c in 0usize..chunks as usize {
            let slot = (plane * chunks as usize + c) * 3;
            p1 += partial[slot];
            p2 += partial[slot + 1];
            p3 += partial[slot + 2];
        }
        let w = weight[channel];
        let sd = std[plane];
        let hw = f32::cast_from(spatial);
        let grad_var = w * p1 * (-0.5f32) * (1.0f32 / (sd * sd * sd));
        let grad_mean = -(w * p2) / sd + grad_var * (-2.0f32 * p3) / hw;
        coefficients[plane * 3] = w / sd;
        coefficients[plane * 3 + 1] = grad_var * 2.0f32 / hw;
        coefficients[plane * 3 + 2] = grad_mean / hw;
        gw = p1 / sd;
        gb = p2;
    }
    let mut scratch = SharedMemory::<f32>::new(2usize);
    let tw = block_sum(gw, &mut scratch, 64usize);
    let tb = block_sum(gb, &mut scratch, 64usize);
    if n == 0 {
        grad_weight[channel] = tw;
        grad_bias[channel] = tb;
    }
}

/// Instance-norm backward pass 3: grad_input = go*a + (x - mean)*b + c.
#[ruda(launch)]
pub fn instance_norm_apply(grad_output: &Array<f32>, x: &Array<f32>, mean: &Array<f32>, coefficients: &Array<f32>,
    grad_input: &mut Array<f32>, spatial: u32, chunk: u32,
) {
    let plane = RUDA_POS_Y as usize;
    let thread = UNIT_POS as usize;
    let m = mean[plane];
    let a = coefficients[plane * 3];
    let b = coefficients[plane * 3 + 1];
    let c = coefficients[plane * 3 + 2];
    let start = RUDA_POS_X as usize * chunk as usize;
    let mut end = start + chunk as usize;
    if end > spatial as usize { end = spatial as usize; }
    let base = plane * spatial as usize;
    let mut i = start + thread;
    while i < end {
        grad_input[base + i] = grad_output[base + i] * a + (x[base + i] - m) * b + c;
        i += 256;
    }
}

// ------------------------------------------------------------------ L1/046

/// Softcapped softmax over the last dim (<= 2048) with the reference bf16
/// rounding points: s = F(x / cap), t = F(tanh s), c = F(t * cap). One warp per row.
#[ruda(launch)]
pub fn softcap_softmax_rows<F: Float>(input: &Array<F>, out: &mut Array<F>, columns: u32, cap: f32,
    #[define(F)] _dtype: StorageType,
) {
    // PyTorch divides by a CPU scalar as a multiply by its float reciprocal.
    let inverse_cap = 1.0f32 / cap;
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS_PLANE as usize;
    let n = columns as usize;
    if row * n < input.len() {
        let mut values = Array::<f32>::new(64usize);
        let mut maximum = f32::new(f32::NEG_INFINITY);
        #[unroll]
        for j in 0usize..64 {
            let col = lane + j * 32;
            let mut v = f32::new(f32::NEG_INFINITY);
            if col < n {
                let scaled = round_to::<F>(f32::cast_from(input[row * n + col]) * inverse_cap);
                let capped = round_to::<F>(round_to::<F>(scaled.tanh()) * cap);
                v = capped;
            }
            values[j] = v;
            if v > maximum { maximum = v; }
        }
        maximum = plane_max(maximum);
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0usize..64 {
            let col = lane + j * 32;
            let mut e = 0.0f32;
            if col < n { e = (values[j] - maximum).exp(); }
            values[j] = e;
            sum += e;
        }
        sum = plane_sum(sum);
        #[unroll]
        for j in 0usize..64 {
            let col = lane + j * 32;
            if col < n { out[row * n + col] = F::cast_from(values[j] / sum); }
        }
    }
}

// ------------------------------------------------------------------ L1/056

/// Mamba dt backward: grad = g * clamp_mask(dt) * sigmoid(dt_bias) (bf16 out)
/// plus per-block head partials of grad. Block = 8 rows x 40 heads.
#[ruda(launch)]
pub fn dt_softplus_clamp_backward<F: Float>(grad_output: &Array<F>, dt_with_bias: &Array<F>, dt_activated: &Array<F>,
    grad_dt: &mut Array<F>, partial: &mut Array<f32>, minimum: f32, maximum: f32,
    #[define(F)] _dtype: StorageType,
) {
    let thread = UNIT_POS as usize;
    let head = thread % 40;
    let lane = thread / 40;
    let rows = grad_output.len() / 40;
    let mut shared = SharedMemory::<f32>::new(320usize);
    let mut total = 0.0f32;
    let mut row = RUDA_POS_X as usize * 8 + lane;
    while row < rows {
        let i = row * 40 + head;
        let activated = f32::cast_from(dt_activated[i]);
        let mut grad = f32::cast_from(grad_output[i]);
        let mut keep = 0.0f32;
        if activated > minimum && activated < maximum { keep = 1.0f32; }
        grad = grad * keep;
        grad = grad * (1.0f32 / (1.0f32 + (-f32::cast_from(dt_with_bias[i])).exp()));
        grad_dt[i] = F::cast_from(grad);
        total += grad;
        row += RUDA_COUNT_X as usize * 8;
    }
    shared[thread] = total;
    sync_ruda();
    if lane == 0 {
        let mut sum = 0.0f32;
        #[unroll]
        for l in 0usize..8 { sum += shared[l * 40 + head]; }
        partial[RUDA_POS_X as usize * 40 + head] = sum;
    }
}

/// Column sum of block partials, rounded to F.
#[ruda(launch)]
pub fn column_sum_cast<F: Float>(partial: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let column = ABSOLUTE_POS as usize;
    let width = out.len();
    if column < width {
        let blocks = partial.len() / width;
        let mut total = 0.0f32;
        for block in 0usize..blocks { total += partial[block * width + column]; }
        out[column] = F::cast_from(total);
    }
}

// ------------------------------------------------------------------ L1/059

/// DeepSeek group-limited routing mask: 8 groups x 32 experts; one warp per
/// token, lane = expert inside a group. Keeps the 4 groups with the largest
/// top-2 sums, writes masked scores (-inf elsewhere) and the group mask.
#[ruda(launch)]
pub fn group_topk_mask(scores: &Array<f32>, masked: &mut Array<f32>, group_mask: &mut Array<f32>) {
    let token = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS_PLANE as usize;
    if token < scores.len() / 256 {
        let mut values = Array::<f32>::new(8usize);
        let mut group = Array::<f32>::new(8usize);
        #[unroll]
        for g in 0usize..8 {
            let v = scores[token * 256 + g * 32 + lane];
            values[g] = v;
            let first = plane_max(v);
            let mut index = 32u32;
            if v == first { index = lane as u32; }
            let first_lane = plane_min(index);
            let mut rest = v;
            if lane as u32 == first_lane { rest = f32::new(f32::NEG_INFINITY); }
            let second = plane_max(rest);
            group[g] = first + second;
        }
        let mut keep = Array::<u32>::new(8usize);
        let mut better = Array::<u32>::new(8usize);
        #[unroll]
        for g in 0usize..8 { better[g] = 0u32; }
        #[unroll]
        for g in 0usize..8 {
            #[unroll]
            for o in 0usize..8 {
                if group[o] > group[g] { better[g] = better[g] + 1u32; }
                if comptime!(o < g) {
                    if group[o] == group[g] { better[g] = better[g] + 1u32; }
                }
            }
        }
        #[unroll]
        for g in 0usize..8 {
            let mut selected = 0u32;
            if better[g] < 4u32 { selected = 1u32; }
            keep[g] = selected;
        }
        #[unroll]
        for g in 0usize..8 {
            let mut v = f32::new(f32::NEG_INFINITY);
            if keep[g] == 1u32 { v = values[g]; }
            masked[token * 256 + g * 32 + lane] = v;
        }
        if lane < 8 {
            let mut flag = 0.0f32;
            #[unroll]
            for g in 0usize..8 { if lane == g && keep[g] == 1u32 { flag = 1.0f32; } }
            group_mask[token * 8 + lane] = flag;
        }
    }
}

// ------------------------------------------------------------------ L1/061

/// Tanh-gated residual backward (hidden 4096, bf16): copies grad_output,
/// writes F(F(g * tanh(gate)) * mask) and per-block partials of
/// sum(g * F(h * mask)). Grid-stride over 16-byte vectors.
#[ruda(launch)]
pub fn tanh_gate_backward<F: Float>(grad_output: &Array<Vector<F, Const<8>>>, gate: &Array<F>,
    hidden: &Array<Vector<F, Const<8>>>, mask: &Array<F>,
    grad_residual: &mut Array<Vector<F, Const<8>>>, grad_hidden: &mut Array<Vector<F, Const<8>>>,
    partial: &mut Array<f32>,
) {
    let gate_value = f32::cast_from(gate[0]).tanh();
    let mut local = 0.0f32;
    let mut v = ABSOLUTE_POS as usize;
    let step = RUDA_COUNT_X as usize * RUDA_DIM_X as usize;
    while v < grad_output.len() {
        let row = v / 512;
        let m = f32::cast_from(mask[row]);
        let g = grad_output[v];
        let h = Vector::<f32, Const<8>>::cast_from(hidden[v]);
        let gf = Vector::<f32, Const<8>>::cast_from(g);
        let mut out = Vector::<F, Const<8>>::empty();
        #[unroll]
        for i in 0usize..8 {
            out[i] = F::cast_from(round_to::<F>(gf[i] * gate_value) * m);
            local += gf[i] * round_to::<F>(h[i] * m);
        }
        grad_residual[v] = g;
        grad_hidden[v] = out;
        v += step;
    }
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let total = block_sum(local, &mut scratch, 256usize);
    if UNIT_POS == 0u32 { partial[RUDA_POS_X as usize] = total; }
}

/// grad_gate = F(sum(partials) * (1 - tanh(gate)^2)).
#[ruda(launch)]
pub fn tanh_gate_finish<F: Float>(partial: &Array<f32>, gate: &Array<F>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let thread = UNIT_POS as usize;
    let mut local = 0.0f32;
    let mut i = thread;
    while i < partial.len() { local += partial[i]; i += 256; }
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let total = block_sum(local, &mut scratch, 256usize);
    if thread == 0 {
        let gate_value = f32::cast_from(gate[0]).tanh();
        out[0] = F::cast_from(total * (1.0f32 - gate_value * gate_value));
    }
}

// ------------------------------------------------------------------ L1/062

/// flag[t] = 1 if t occurs in positions.
#[ruda(launch)]
pub fn position_flags(positions: &Array<i64>, flags: &mut Array<u32>) {
    let t = ABSOLUTE_POS as usize;
    if t < flags.len() {
        let mut hit = 0u32;
        for j in 0usize..positions.len() {
            if positions[j] == i64::cast_from(t) { hit = 1u32; }
        }
        flags[t] = hit;
    }
}

/// Copy key/value cache gradients, zeroing rows whose position is flagged.
/// 16 vectors of 8 bf16 per 128-wide row.
#[ruda(launch)]
pub fn cache_copy_zero_rows<F: Float>(key: &Array<Vector<F, Const<8>>>, value: &Array<Vector<F, Const<8>>>,
    flags: &Array<u32>, key_out: &mut Array<Vector<F, Const<8>>>, value_out: &mut Array<Vector<F, Const<8>>>,
    #[define(F)] _dtype: StorageType,
) {
    let mut v = ABSOLUTE_POS as usize;
    let step = RUDA_COUNT_X as usize * RUDA_DIM_X as usize;
    let length = flags.len();
    while v < key.len() {
        let t = (v / 16) % length;
        if flags[t] == 1u32 {
            key_out[v] = Vector::<F, Const<8>>::new(F::new(0.0));
            value_out[v] = Vector::<F, Const<8>>::new(F::new(0.0));
        } else {
            key_out[v] = key[v];
            value_out[v] = value[v];
        }
        v += step;
    }
}

/// KV-cache RoPE backward per (batch, new position): 64 threads own the
/// dimension pairs (d, d + 64) and loop over the 8 KV heads with bf16
/// rounding after every elementwise op as in the reference.
#[ruda(launch)]
pub fn kv_rope_backward<F: Float>(grad_key_cache: &Array<F>, grad_value_cache: &Array<F>, key_states: &Array<F>,
    cos: &Array<F>, sin: &Array<F>, positions: &Array<i64>,
    grad_key: &mut Array<F>, grad_value: &mut Array<F>, grad_cos: &mut Array<F>, grad_sin: &mut Array<F>,
    cache_length: u32, #[define(F)] _dtype: StorageType,
) {
    let j = RUDA_POS_X as usize;
    let b = RUDA_POS_Y as usize;
    let d = UNIT_POS as usize;
    let fresh = positions.len();
    let t = u32::cast_from(positions[j]) as usize;
    let rope = (b * fresh + j) * 128;
    let c0 = f32::cast_from(cos[rope + d]);
    let c1 = f32::cast_from(cos[rope + d + 64]);
    let s0 = f32::cast_from(sin[rope + d]);
    let s1 = f32::cast_from(sin[rope + d + 64]);
    let mut cos0 = 0.0f32;
    let mut cos1 = 0.0f32;
    let mut sin0 = 0.0f32;
    let mut sin1 = 0.0f32;
    #[unroll]
    for h in 0usize..8 {
        let cache_row = ((b * 8 + h) * cache_length as usize + t) * 128;
        let state_row = ((b * 8 + h) * fresh + j) * 128;
        let g0 = f32::cast_from(grad_key_cache[cache_row + d]);
        let g1 = f32::cast_from(grad_key_cache[cache_row + d + 64]);
        let k0 = f32::cast_from(key_states[state_row + d]);
        let k1 = f32::cast_from(key_states[state_row + d + 64]);
        let from_cos0 = round_to::<F>(g0 * c0);
        let from_cos1 = round_to::<F>(g1 * c1);
        let rotated0 = round_to::<F>(g0 * s0);
        let rotated1 = round_to::<F>(g1 * s1);
        grad_key[state_row + d] = F::cast_from(from_cos0 + rotated1);
        grad_key[state_row + d + 64] = F::cast_from(from_cos1 + (-rotated0));
        grad_value[state_row + d] = grad_value_cache[cache_row + d];
        grad_value[state_row + d + 64] = grad_value_cache[cache_row + d + 64];
        cos0 += round_to::<F>(g0 * k0);
        cos1 += round_to::<F>(g1 * k1);
        sin0 += round_to::<F>(g0 * (-k1));
        sin1 += round_to::<F>(g1 * k0);
    }
    grad_cos[rope + d] = F::cast_from(cos0);
    grad_cos[rope + d + 64] = F::cast_from(cos1);
    grad_sin[rope + d] = F::cast_from(sin0);
    grad_sin[rope + d + 64] = F::cast_from(sin1);
}

// ------------------------------------------------------------------ L1/066

/// Masked softmax backward with optional dropout rescale (fp32, rows <= 4096,
/// 256 threads x 16 columns cached per row). mask is broadcast over heads.
#[ruda(launch)]
pub fn softmax_dropout_backward(grad_output: &Array<f32>, probabilities: &Array<f32>, mask: &Array<u8>,
    dropout: &Array<u8>, out: &mut Array<f32>, columns: u32, heads: u32, use_dropout: u32, inverse_keep: f32,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = columns as usize;
    let query = row % n;
    let batch = row / (n * heads as usize);
    let mut p = Array::<f32>::new(16usize);
    let mut g = Array::<f32>::new(16usize);
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..16 {
        let col = thread + j * 256;
        let mut pv = 0.0f32;
        let mut gv = 0.0f32;
        if col < n {
            pv = probabilities[row * n + col];
            gv = grad_output[row * n + col];
            if use_dropout == 1u32 { gv = (gv * f32::cast_from(dropout[row * n + col])) * inverse_keep; }
            local += pv * gv;
        }
        p[j] = pv;
        g[j] = gv;
    }
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let sum = block_sum(local, &mut scratch, 256usize);
    #[unroll]
    for j in 0usize..16 {
        let col = thread + j * 256;
        if col < n {
            let mut value = p[j] * (g[j] - sum);
            if mask[(batch * n + query) * n + col] == 0u8 { value = 0.0f32; }
            out[row * n + col] = value;
        }
    }
}

// ------------------------------------------------------------------ L1/068

/// RoPE application backward with GQA (fp32, head_dim 128): one block of 64
/// threads per (batch, seq); thread d owns (d, d + 64) for every q / kv head.
#[ruda(launch)]
pub fn rope_gqa_backward(grad_q_embed: &Array<f32>, grad_k_embed: &Array<f32>, q: &Array<f32>, k: &Array<f32>,
    cos: &Array<f32>, sin: &Array<f32>, grad_q: &mut Array<f32>, grad_k: &mut Array<f32>,
    grad_cos: &mut Array<f32>, grad_sin: &mut Array<f32>, seq: u32, heads: u32, kv_heads: u32,
) {
    let row = RUDA_POS_X as usize;
    let d = UNIT_POS as usize;
    let s = row % seq as usize;
    let b = row / seq as usize;
    let c0 = cos[row * 128 + d];
    let c1 = cos[row * 128 + d + 64];
    let s0 = sin[row * 128 + d];
    let s1 = sin[row * 128 + d + 64];
    let mut qc0 = 0.0f32;
    let mut qc1 = 0.0f32;
    let mut qs0 = 0.0f32;
    let mut qs1 = 0.0f32;
    for h in 0usize..heads as usize {
        let base = ((b * heads as usize + h) * seq as usize + s) * 128;
        let g0 = grad_q_embed[base + d];
        let g1 = grad_q_embed[base + d + 64];
        let x0 = q[base + d];
        let x1 = q[base + d + 64];
        grad_q[base + d] = g0 * c0 - (-g1) * s0;
        grad_q[base + d + 64] = g1 * c1 - g0 * s1;
        qc0 += g0 * x0;
        qc1 += g1 * x1;
        qs0 += g0 * (-x1);
        qs1 += g1 * x0;
    }
    let mut kc0 = 0.0f32;
    let mut kc1 = 0.0f32;
    let mut ks0 = 0.0f32;
    let mut ks1 = 0.0f32;
    for h in 0usize..kv_heads as usize {
        let base = ((b * kv_heads as usize + h) * seq as usize + s) * 128;
        let g0 = grad_k_embed[base + d];
        let g1 = grad_k_embed[base + d + 64];
        let x0 = k[base + d];
        let x1 = k[base + d + 64];
        grad_k[base + d] = g0 * c0 - (-g1) * s0;
        grad_k[base + d + 64] = g1 * c1 - g0 * s1;
        kc0 += g0 * x0;
        kc1 += g1 * x1;
        ks0 += g0 * (-x1);
        ks1 += g1 * x0;
    }
    grad_cos[row * 128 + d] = qc0 + kc0;
    grad_cos[row * 128 + d + 64] = qc1 + kc1;
    grad_sin[row * 128 + d] = qs0 + ks0;
    grad_sin[row * 128 + d + 64] = qs1 + ks1;
}

// ------------------------------------------------------------------ L1/071

/// cat(cache, rope(new keys)) and cat(value cache, new values) along seq
/// (bf16, head_dim 128 = 16 vectors of 8). RoPE in bf16 arithmetic.
#[ruda(launch)]
pub fn kv_cache_append_rope<F: Float>(key_states: &Array<Vector<F, Const<8>>>, value_states: &Array<Vector<F, Const<8>>>,
    cos: &Array<Vector<F, Const<8>>>, sin: &Array<Vector<F, Const<8>>>,
    key_cache: &Array<Vector<F, Const<8>>>, value_cache: &Array<Vector<F, Const<8>>>,
    key_out: &mut Array<Vector<F, Const<8>>>, value_out: &mut Array<Vector<F, Const<8>>>,
    current: u32, fresh: u32, heads: u32, #[define(F)] _dtype: StorageType,
) {
    let total = current as usize + fresh as usize;
    let mut v = ABSOLUTE_POS as usize;
    let step = RUDA_COUNT_X as usize * RUDA_DIM_X as usize;
    while v < key_out.len() {
        let chunk = v % 16;
        let row = v / 16;
        let t = row % total;
        let bh = row / total;
        if t < current as usize {
            let source = (bh * current as usize + t) * 16 + chunk;
            key_out[v] = key_cache[source];
            value_out[v] = value_cache[source];
        } else {
            let j = t - current as usize;
            let state = (bh * fresh as usize + j) * 16;
            let b = bh / heads as usize;
            let rope = (b * fresh as usize + j) * 16;
            let x = Vector::<f32, Const<8>>::cast_from(key_states[state + chunk]);
            let mut partner = Vector::<f32, Const<8>>::empty();
            if chunk < 8 {
                partner = -Vector::<f32, Const<8>>::cast_from(key_states[state + chunk + 8]);
            } else {
                partner = Vector::<f32, Const<8>>::cast_from(key_states[state + chunk - 8]);
            }
            let c = Vector::<f32, Const<8>>::cast_from(cos[rope + chunk]);
            let s = Vector::<f32, Const<8>>::cast_from(sin[rope + chunk]);
            let mut out = Vector::<F, Const<8>>::empty();
            #[unroll]
            for i in 0usize..8 {
                out[i] = F::cast_from(round_to::<F>(x[i] * c[i]) + round_to::<F>(partner[i] * s[i]));
            }
            key_out[v] = out;
            value_out[v] = value_states[state + chunk];
        }
        v += step;
    }
}

// ------------------------------------------------------------------ L1/078

/// GroupNorm pass 1: f64 (sum, sum of squares) per (batch, channel, chunk).
/// Grid (chunks, channels, batch), 256 threads.
#[ruda(launch)]
pub fn group_norm_partial(x: &Array<f32>, partial: &mut Array<f64>, spatial: u32, chunk: u32) {
    let chunk_index = RUDA_POS_X as usize;
    let channel = RUDA_POS_Y as usize;
    let batch = RUDA_POS_Z as usize;
    let channels = RUDA_COUNT_Y as usize;
    let chunks = RUDA_COUNT_X as usize;
    let base = (batch * channels + channel) * spatial as usize;
    let start = chunk_index * chunk as usize;
    let mut end = start + chunk as usize;
    if end > spatial as usize { end = spatial as usize; }
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    let mut i = start + UNIT_POS as usize;
    while i < end {
        let value = f64::cast_from(x[base + i]);
        sum += value;
        squares += value * value;
        i += 256;
    }
    let mut shared = SharedMemory::<f64>::new(512usize);
    let thread = UNIT_POS as usize;
    shared[thread] = sum;
    shared[256 + thread] = squares;
    sync_ruda();
    let mut width = 128usize;
    while width > 0 {
        if thread < width {
            shared[thread] += shared[thread + width];
            shared[256 + thread] += shared[256 + thread + width];
        }
        sync_ruda();
        width /= 2;
    }
    if thread == 0 {
        let slot = ((batch * channels + channel) * chunks + chunk_index) * 2;
        partial[slot] = shared[0];
        partial[slot + 1] = shared[256];
    }
}

/// GroupNorm pass 2: merge the group's partials, then
/// out = ((x - mean) / sqrt(var + eps)) * w + b for this block's chunk.
#[ruda(launch)]
pub fn group_norm_apply(x: &Array<f32>, partial: &Array<f64>, weight: &Array<f32>, bias: &Array<f32>,
    out: &mut Array<f32>, spatial: u32, chunk: u32, groups: u32, eps: f32,
) {
    let chunk_index = RUDA_POS_X as usize;
    let channel = RUDA_POS_Y as usize;
    let batch = RUDA_POS_Z as usize;
    let channels = RUDA_COUNT_Y as usize;
    let chunks = RUDA_COUNT_X as usize;
    let per_group = channels / groups as usize;
    let first = (channel / per_group) * per_group;
    let thread = UNIT_POS as usize;
    let mut shared = SharedMemory::<f64>::new(512usize);
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    let mut p = thread;
    while p < per_group * chunks {
        let slot = ((batch * channels + first + p / chunks) * chunks + p % chunks) * 2;
        sum += partial[slot];
        squares += partial[slot + 1];
        p += 256;
    }
    shared[thread] = sum;
    shared[256 + thread] = squares;
    sync_ruda();
    let mut width = 128usize;
    while width > 0 {
        if thread < width {
            shared[thread] += shared[thread + width];
            shared[256 + thread] += shared[256 + thread + width];
        }
        sync_ruda();
        width /= 2;
    }
    let count = f64::cast_from(per_group * spatial as usize);
    let mean64 = shared[0] / count;
    let mut variance64 = shared[256] / count - mean64 * mean64;
    if variance64 < 0.0f64 { variance64 = 0.0f64; }
    let mean = f32::cast_from(mean64);
    let deviation = (f32::cast_from(variance64) + eps).sqrt();
    let w = weight[channel];
    let b = bias[channel];
    let base = (batch * channels + channel) * spatial as usize;
    let start = chunk_index * chunk as usize;
    let mut end = start + chunk as usize;
    if end > spatial as usize { end = spatial as usize; }
    let mut i = start + thread;
    while i < end {
        out[base + i] = ((x[base + i] - mean) / deviation) * w + b;
        i += 256;
    }
}

// ------------------------------------------------------------------ L1/084

/// SiLU backward from a precomputed sigmoid (fp32, any length).
#[ruda(launch)]
pub fn silu_backward_from_sigmoid(grad_output: &Array<f32>, x: &Array<f32>, sigmoid: &Array<f32>, out: &mut Array<f32>) {
    let base = RUDA_POS_X as usize * 1024 + UNIT_POS as usize;
    #[unroll]
    for j in 0usize..4 {
        let i = base + j * 256;
        if i < out.len() {
            let s = sigmoid[i];
            let bracket = 1.0f32 + x[i] * (1.0f32 - s);
            out[i] = grad_output[i] * (s * bracket);
        }
    }
}

// ------------------------------------------------------------------ L1/085

/// GEGLU with tanh GELU (fp32): out = gelu(x[:, :inner]) * x[:, inner:].
#[ruda(launch)]
pub fn geglu_tanh_f32(x: &Array<Vector<f32, Const<4>>>, out: &mut Array<Vector<f32, Const<4>>>, #[comptime] inner: usize) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let quarter = comptime!(inner / 4);
        let row = p / quarter;
        let column = p % quarter;
        let gate = x[row * 2 * quarter + column];
        let linear = x[row * 2 * quarter + quarter + column];
        let mut result = Vector::<f32, Const<4>>::empty();
        #[unroll]
        for i in 0usize..4 {
            let value = gate[i];
            let cubed = value * value * value;
            let inner_value = 0.7978845608028654f32 * (value + 0.044715f32 * cubed);
            result[i] = (0.5f32 * value * (1.0f32 + inner_value.tanh())) * linear[i];
        }
        out[p] = result;
    }
}

// ------------------------------------------------------------------ L1/088

/// RoPE application (fp32, head_dim 128 = 32 vectors of 4); RUDA_POS_Y
/// selects q (0) or k (1). cos/sin are (seq, 128).
#[ruda(launch)]
pub fn rope_apply_f32(query: &Array<Vector<f32, Const<4>>>, key: &Array<Vector<f32, Const<4>>>,
    cos: &Array<Vector<f32, Const<4>>>, sin: &Array<Vector<f32, Const<4>>>,
    query_out: &mut Array<Vector<f32, Const<4>>>, key_out: &mut Array<Vector<f32, Const<4>>>, seq: u32,
) {
    let mut v = ABSOLUTE_POS as usize;
    let step = RUDA_COUNT_X as usize * RUDA_DIM_X as usize;
    let is_key = RUDA_POS_Y == 1u32;
    let mut length = query.len();
    if is_key { length = key.len(); }
    while v < length {
        let chunk = v % 32;
        let row = v / 32;
        let s = row % seq as usize;
        let mut x = query[v];
        let mut other = query[v - chunk + (chunk + 16) % 32];
        if is_key {
            x = key[v];
            other = key[v - chunk + (chunk + 16) % 32];
        }
        if chunk < 16 { other = -other; }
        let result = x * cos[s * 32 + chunk] + other * sin[s * 32 + chunk];
        if is_key { key_out[v] = result; } else { query_out[v] = result; }
        v += step;
    }
}

// ------------------------------------------------------------------ L1/090

/// d(theta) = -grad_cos * sin(theta) + grad_sin * cos(theta) (bf16 grads, fp32 out).
#[ruda(launch)]
pub fn rope_theta_backward<F: Float>(grad_cos: &Array<Vector<F, Const<4>>>, grad_sin: &Array<Vector<F, Const<4>>>,
    theta: &Array<Vector<f32, Const<4>>>, out: &mut Array<Vector<f32, Const<4>>>, #[define(F)] _dtype: StorageType,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let gc = Vector::<f32, Const<4>>::cast_from(grad_cos[p]);
        let gs = Vector::<f32, Const<4>>::cast_from(grad_sin[p]);
        let t = theta[p];
        let mut result = Vector::<f32, Const<4>>::empty();
        #[unroll]
        for i in 0usize..4 { result[i] = -gc[i] * t[i].sin() + gs[i] * t[i].cos(); }
        out[p] = result;
    }
}

// ------------------------------------------------------------------ L1/094

/// RWKV-style time-decay recurrence: one thread per (batch, channel) walks the
/// sequence with the next step's key/value prefetched.
#[ruda(launch)]
pub fn time_decay_scan(time_decay: &Array<f32>, key: &Array<f32>, time_first: &Array<f32>, value: &Array<f32>,
    max_in: &Array<f32>, num_in: &Array<f32>, den_in: &Array<f32>,
    out: &mut Array<f32>, max_out: &mut Array<f32>, num_out: &mut Array<f32>, den_out: &mut Array<f32>,
    seq: u32,
) {
    let index = ABSOLUTE_POS as usize;
    let hidden = time_decay.len();
    if index < max_in.len() {
        let c = index % hidden;
        let b = index / hidden;
        let decay = -time_decay[c].exp();
        let first = time_first[c];
        let mut max_state = max_in[index];
        let mut num_state = num_in[index];
        let mut den_state = den_in[index];
        let mut position = b * seq as usize * hidden + c;
        let mut k_next = key[position];
        let mut v_next = value[position];
        for t in 0usize..seq as usize {
            let k = k_next;
            let v = v_next;
            if t + 1 < seq as usize {
                k_next = key[position + hidden];
                v_next = value[position + hidden];
            }
            let shifted = k + first;
            let mut max_output = max_state;
            if shifted > max_output { max_output = shifted; }
            let e1 = (max_state - max_output).exp();
            let e2 = (shifted - max_output).exp();
            out[position] = (e1 * num_state + e2 * v) / (e1 * den_state + e2);
            let decayed = max_state + decay;
            let mut max_state_next = decayed;
            if k > max_state_next { max_state_next = k; }
            let s1 = (decayed - max_state_next).exp();
            let s2 = (k - max_state_next).exp();
            num_state = s1 * num_state + s2 * v;
            den_state = s1 * den_state + s2;
            max_state = max_state_next;
            position += hidden;
        }
        max_out[index] = max_state;
        num_out[index] = num_state;
        den_out[index] = den_state;
    }
}
