//! L1 batch C kernels (problems 001, 005, 006, 008, 009) plus shared row helpers.
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

use crate::standalone::block_sum;
use super::common::rnd;

/// Softmax backward rows with optional dropout: g = dp * keep * inv_keep,
/// out = F(p * (g - sum(g * p))). dp rows have stride `ld`.
#[ruda(launch)]
pub fn softmax_bwd_rows<F: Float>(dp: &Array<f32>, p: &Array<F>, keep: &Array<u8>, out: &mut Array<F>,
    columns: u32, ld: u32, use_dropout: u32, inv_keep: f32, #[define(F)] _f: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = columns as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < n {
        let mut g = dp[row * ld as usize + j];
        if use_dropout != 0u32 { g = g * f32::cast_from(keep[row * n + j]) * inv_keep; }
        local += g * f32::cast_from(p[row * n + j]);
        j += 256;
    }
    let total = block_sum(local, &mut scratch, 256usize);
    let mut j2 = thread;
    while j2 < n {
        let mut g = dp[row * ld as usize + j2];
        if use_dropout != 0u32 { g = g * f32::cast_from(keep[row * n + j2]) * inv_keep; }
        let pv = f32::cast_from(p[row * n + j2]);
        out[row * n + j2] = F::cast_from(pv * (g - total));
        j2 += 256;
    }
}

/// Zero-padded row copy (rows, columns) -> (rows, ld).
#[ruda(launch)]
pub fn pad_rows<F: Float>(x: &Array<F>, out: &mut Array<F>, count: u32, columns: u32, ld: u32, #[define(F)] _f: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let c = i % ld as usize;
        let r = i / ld as usize;
        let mut v = F::cast_from(0.0f32);
        if c < columns as usize { v = x[r * columns as usize + c]; }
        out[i] = v;
    }
}

/// L1/005: y[t, c] = C * conv(B * x)[t, c], causal depthwise conv (k = 4) over
/// time with bias; BCx rows are (B, C, x) chunks of 3H. bf16 rounding per op.
#[ruda(launch)]
pub fn conv_gate<F: Float>(bcx: &Array<F>, weight: &Array<F>, bias: &Array<F>, out: &mut Array<F>,
    count: u32, seq: u32, hidden: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let h = hidden as usize;
        let c = i % h;
        let row = i / h;
        let t = row % seq as usize;
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0usize..4 {
            if t + k >= 3 {
                let source = row + k - 3;
                let bx = rnd::<F>(f32::cast_from(bcx[source * 3 * h + c]) * f32::cast_from(bcx[source * 3 * h + 2 * h + c]));
                acc += f32::cast_from(weight[c * 4 + k]) * bx;
            }
        }
        let conv = rnd::<F>(acc + f32::cast_from(bias[c]));
        out[i] = F::cast_from(f32::cast_from(bcx[row * 3 * h + h + c]) * conv);
    }
}

/// L1/006: depthwise conv1d (k = 3, left pad 2) on (B, 768, L), returns
/// v * x0, x0, x1 for the three 256-channel groups.
#[ruda(launch)]
pub fn hyena_short_conv(u: &Array<f32>, weight: &Array<f32>, bias: &Array<f32>,
    gated: &mut Array<f32>, x0: &mut Array<f32>, x1: &mut Array<f32>, count: u32, length: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let l = length as usize;
        let t = i % l;
        let c = (i / l) % 256;
        let b = i / (l * 256);
        let mut values = Array::<f32>::new(3usize);
        #[unroll]
        for g in 0usize..3 {
            let channel = g * 256 + c;
            let base = (b * 768 + channel) * l;
            let mut acc = bias[channel];
            #[unroll]
            for k in 0usize..3 {
                if t + k >= 2 { acc += weight[channel * 3 + k] * u[base + t + k - 2]; }
            }
            values[g] = acc;
        }
        gated[i] = values[2] * values[0];
        x0[i] = values[0];
        x1[i] = values[1];
    }
}

/// acc[index[r], :] += source[r, :] (f32 atomics).
#[ruda(launch)]
pub fn index_add_rows<F: Float>(source: &Array<F>, index: &Array<i64>, acc: &mut Array<Atomic<f32>>,
    count: u32, width: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let r = i / width as usize;
        let c = i % width as usize;
        let target = index[r] as usize * width as usize + c;
        acc[target].fetch_add(f32::cast_from(source[i]));
    }
}

/// out[index[r], :] = O(source[r, :]) for rows of 8-element vectors.
#[ruda(launch)]
pub fn scatter_rows<F: Float, O: Float>(source: &Array<Vector<F, Const<8>>>, index: &Array<i64>,
    out: &mut Array<Vector<O, Const<8>>>, count: u32, vectors: u32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let r = i / vectors as usize;
        let target = index[r] as usize;
        out[target * vectors as usize + i % vectors as usize] = Vector::<O, Const<8>>::cast_from(source[i]);
    }
}

/// L1/009 prep (one block of 256 per selected token): g = grad[index],
/// gsw = sum(g * expert_out) -> routing[index] (bf16), gE = bf16(g * weight).
#[ruda(launch)]
pub fn moe_scatter_prep<F: Float>(grad: &Array<Vector<F, Const<8>>>, index: &Array<i64>, expert: &Array<Vector<F, Const<8>>>,
    weights: &Array<F>, routing: &mut Array<F>, grad_expert: &mut Array<Vector<F, Const<8>>>, vectors: u32,
    #[define(F)] _f: StorageType,
) {
    let r = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let token = index[r] as usize;
    let w = f32::cast_from(weights[r]);
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < vectors as usize {
        let g = Vector::<f32, Const<8>>::cast_from(grad[token * vectors as usize + j]);
        let e = Vector::<f32, Const<8>>::cast_from(expert[r * vectors as usize + j]);
        #[unroll]
        for d in 0usize..8 { local += g[d] * e[d]; }
        grad_expert[r * vectors as usize + j] = Vector::<F, Const<8>>::cast_from(g * Vector::new(w));
        j += 256;
    }
    let total = block_sum(local, &mut scratch, 256usize);
    if thread == 0 { routing[token] = F::cast_from(total); }
}

/// L1/009 middle: from G = gE @ W2 (f32): gu = bf16(G * gate), gw1 = bf16(G * up * dsilu(w1)).
#[ruda(launch)]
pub fn moe_swiglu_bwd<F: Float>(g: &Array<f32>, gate: &Array<F>, up: &Array<F>, w1: &Array<F>,
    grad_up: &mut Array<F>, grad_w1: &mut Array<F>, count: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let gg = g[i];
        let x = f32::cast_from(w1[i]);
        let s = 1.0f32 / (1.0f32 + (-x).exp());
        grad_up[i] = F::cast_from(gg * f32::cast_from(gate[i]));
        grad_w1[i] = F::cast_from(gg * f32::cast_from(up[i]) * (s * (1.0f32 + x * (1.0f32 - s))));
    }
}

/// L1/017: from G = dO @ Wd (f32): grad_up = bf16(G * gate_silu),
/// grad_gate = bf16(G * up * sig * (1 + gate * (1 - sig))) with the given sigmoid.
#[ruda(launch)]
pub fn swiglu_bwd_sig<F: Float>(g: &Array<f32>, up: &Array<F>, gate_silu: &Array<F>, gate: &Array<F>, sigmoid: &Array<F>,
    grad_up: &mut Array<F>, grad_gate: &mut Array<F>, count: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let gg = g[i];
        let s = f32::cast_from(sigmoid[i]);
        grad_up[i] = F::cast_from(gg * f32::cast_from(gate_silu[i]));
        grad_gate[i] = F::cast_from(gg * f32::cast_from(up[i]) * s * (1.0f32 + f32::cast_from(gate[i]) * (1.0f32 - s)));
    }
}

/// L1/060: softmax backward from upstream grad g (F): out = F(F(p * (g - sum(g * p))) * scale),
/// written with row stride `ld` (zero padded beyond `columns`).
#[ruda(launch)]
pub fn softmax_bwd_scaled<F: Float>(g: &Array<F>, p: &Array<F>, out: &mut Array<F>, columns: u32, ld: u32, scale: f32,
    #[define(F)] _f: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = columns as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < n { local += f32::cast_from(g[row * n + j]) * f32::cast_from(p[row * n + j]); j += 256; }
    let total = block_sum(local, &mut scratch, 256usize);
    let mut j2 = thread;
    while j2 < ld as usize {
        let mut v = 0.0f32;
        if j2 < n {
            let pv = f32::cast_from(p[row * n + j2]);
            v = rnd::<F>(rnd::<F>(pv * (f32::cast_from(g[row * n + j2]) - total)) * scale);
        }
        out[row * ld as usize + j2] = F::cast_from(v);
        j2 += 256;
    }
}
