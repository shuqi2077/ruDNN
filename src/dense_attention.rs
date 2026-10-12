//! Dense SDPA with plane-cooperative online softmax and recomputed backward.
//! Scores, probabilities and dropout masks are never stored as L x S tensors.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn mask_offset(mask: &Tensor<f32>, position: usize) -> usize {
    let mut remaining = position;
    let mut offset = 0usize;
    let mut dim = mask.rank();
    while dim > 0 {
        dim -= 1;
        offset += (remaining % mask.shape(dim)) * mask.stride(dim);
        remaining /= mask.shape(dim);
    }
    offset
}

#[ruda]
fn keep(position: usize, seed_low: u32, seed_high: u32, base_low: u32, base_high: u32, p: f32) -> f32 {
    let block = u32::cast_from(position / 4);
    let mut a = base_low + block;
    let mut b = base_high;
    if a < base_low { b += 1; }
    let mut c = 0u32;
    let mut d = 0u32;
    let mut ka = seed_low;
    let mut kb = seed_high;
    for _round in 0..10 {
        let ha = a.mul_hi(0xD2511F53u32);
        let hc = c.mul_hi(0xCD9E8D57u32);
        let la = a * 0xD2511F53u32;
        let lc = c * 0xCD9E8D57u32;
        a = hc ^ b ^ ka; b = lc; c = ha ^ d ^ kb; d = la;
        ka += 0x9E3779B9u32; kb += 0xBB67AE85u32;
    }
    let lane = position % 4;
    let mut word = a;
    if lane == 1 { word = b; } else if lane == 2 { word = c; } else if lane == 3 { word = d; }
    let unit = f32::cast_from(word >> 8) * (1.0f32 / 16777216.0f32);
    let mut result = 0.0f32;
    if p < 1.0 && unit >= p { result = 1.0 / (1.0 - p); }
    result
}

#[ruda(launch)]
pub fn forward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, mask: &Tensor<f32>,
    out: &mut Array<F>, stats: &mut Array<f32>,
    heads: u32, kv_heads: u32, queries: u32, keys: u32, scale: f32, p: f32,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] lanes: usize, #[comptime] causal: bool, #[comptime] mask_kind: u32,
    #[comptime] dropout: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let head = (row / queries as usize) % heads as usize;
    let batch = row / (heads as usize * queries as usize);
    let kh = head / (heads / kv_heads) as usize;
    let mut query = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut acc = Array::<f32>::new((value_dim + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        query[j] = 0.0;
        if d < key_dim { query[j] = f32::cast_from(q[row * key_dim + d]); }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) { acc[j] = 0.0; }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    for token in 0usize..keys as usize {
        let cache_row = (batch * kv_heads as usize + kh) * keys as usize + token;
        let mut partial = 0.0f32;
        #[unroll]
        for j in 0usize..((key_dim + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < key_dim { partial = fma(query[j], f32::cast_from(k[cache_row * key_dim + d]), partial); }
        }
        let mut score = plane_sum(partial) * scale;
        let mut allowed = true;
        if comptime!(causal) { allowed = token <= row % queries as usize; }
        if comptime!(mask_kind != 0) {
            let m = mask[mask_offset(mask, row * keys as usize + token)];
            if comptime!(mask_kind == 1) { allowed = m != 0.0; }
            else { score += m; }
        }
        // Additive -inf excludes a score, matching the public masked_softmax.
        if comptime!(mask_kind == 2) { allowed = score != f32::NEG_INFINITY; }
        let mut weight = 0.0f32;
        let mut old_scale = 1.0f32;
        if allowed {
            let mut next_max = score;
            if denominator != 0.0 { next_max = f32::max(maximum, score); }
            old_scale = 0.0;
            if denominator != 0.0 { old_scale = (maximum - next_max).exp(); }
            weight = (score - next_max).exp();
            denominator = fma(denominator, old_scale, weight);
            maximum = next_max;
        }
        let mut multiplier = 1.0f32;
        if comptime!(dropout) { multiplier = keep(row * keys as usize + token, seed_low, seed_high, base_low, base_high, p); }
        let mut dropped_weight = weight * multiplier;
        if comptime!(dropout) { if p == 1.0 { dropped_weight = 0.0; } }
        let mut accumulator_scale = old_scale;
        if comptime!(dropout) { if p == 1.0 { accumulator_scale = 1.0; } }
        // Masked zero weights still multiply V, preserving IEEE NaN propagation.
        #[unroll]
        for j in 0usize..((value_dim + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < value_dim { acc[j] = fma(dropped_weight, f32::cast_from(v[cache_row * value_dim + d]), acc[j] * accumulator_scale); }
        }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < value_dim {
            let mut result = acc[j];
            if denominator != 0.0 && p != 1.0 { result /= denominator; }
            out[row * value_dim + d] = F::cast_from(result);
        }
    }
    if lane == 0 { stats[row * 2] = maximum; stats[row * 2 + 1] = denominator; }
}

#[ruda(launch)]
pub fn backward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, mask: &Tensor<f32>,
    grad: &Array<F>, stats: &Array<f32>, dq: &mut Array<F>,
    dk: &mut Array<Atomic<f32>>, dv: &mut Array<Atomic<f32>>, dm: &mut Array<Atomic<f32>>,
    heads: u32, kv_heads: u32, queries: u32, keys: u32, scale: f32, p: f32,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] mask_kind: u32, #[comptime] dropout: bool,
    #[comptime] need_mask: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let head = (row / queries as usize) % heads as usize;
    let batch = row / (heads as usize * queries as usize);
    let kh = head / (heads / kv_heads) as usize;
    let maximum = stats[row * 2];
    let denominator = stats[row * 2 + 1];
    let mut query = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut qgrad = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut dy = Array::<f32>::new((value_dim + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes; query[j] = 0.0; qgrad[j] = 0.0;
        if d < key_dim { query[j] = f32::cast_from(q[row * key_dim + d]); }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes; dy[j] = 0.0;
        if d < value_dim { dy[j] = f32::cast_from(grad[row * value_dim + d]); }
    }
    let mut expected = 0.0f32;
    // First scan computes the exact FP32 softmax Jacobian contraction. Using
    // low-precision saved O here would introduce an additional rounding error.
    for pass in 0usize..2 {
        for token in 0usize..keys as usize {
            let cache_row = (batch * kv_heads as usize + kh) * keys as usize + token;
            let mut partial = 0.0f32;
            let mut dp_local = 0.0f32;
            #[unroll]
            for j in 0usize..((key_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < key_dim { partial = fma(query[j], f32::cast_from(k[cache_row * key_dim + d]), partial); }
            }
            #[unroll]
            for j in 0usize..((value_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < value_dim { dp_local = fma(dy[j], f32::cast_from(v[cache_row * value_dim + d]), dp_local); }
            }
            let mut score = plane_sum(partial) * scale;
            let mut allowed = true;
            if comptime!(causal) { allowed = token <= row % queries as usize; }
            let mut mi = 0usize;
            if comptime!(mask_kind != 0) {
                mi = mask_offset(mask, row * keys as usize + token);
                let m = mask[mi];
                if comptime!(mask_kind == 1) { allowed = m != 0.0; } else { score += m; }
            }
            if comptime!(mask_kind == 2) { allowed = score != f32::NEG_INFINITY; }
            let mut probability = 0.0f32;
            if allowed && denominator != 0.0 { probability = (score - maximum).exp() / denominator; }
            let mut multiplier = 1.0f32;
            if comptime!(dropout) { multiplier = keep(row * keys as usize + token, seed_low, seed_high, base_low, base_high, p); }
            let mut dp = plane_sum(dp_local) * multiplier;
            if comptime!(dropout) { if p == 1.0 { dp = 0.0; } }
            if pass == 0 { expected += probability * dp; }
            else {
                let mut ds = 0.0f32;
                if allowed { ds = probability * (dp - expected); }
                #[unroll]
                for j in 0usize..((key_dim + lanes - 1) / lanes) {
                    let d = lane + j * lanes;
                    if d < key_dim {
                        qgrad[j] = fma(ds * scale, f32::cast_from(k[cache_row * key_dim + d]), qgrad[j]);
                        dk[cache_row * key_dim + d].fetch_add(ds * scale * query[j]);
                    }
                }
                #[unroll]
                for j in 0usize..((value_dim + lanes - 1) / lanes) {
                    let d = lane + j * lanes;
                    let mut dropped_probability = probability * multiplier;
                    if comptime!(dropout) { if p == 1.0 { dropped_probability = 0.0; } }
                    if d < value_dim { dv[cache_row * value_dim + d].fetch_add(dropped_probability * dy[j]); }
                }
                if comptime!(need_mask) { if lane == 0 { dm[mi].fetch_add(ds); } }
            }
        }
    }
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < key_dim { dq[row * key_dim + d] = F::cast_from(qgrad[j]); }
    }
}

#[ruda(launch)]
pub fn zero(out: &mut Array<f32>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() { out[pos] = 0.0; }
}

#[ruda(launch)]
pub fn narrow<F: Float>(input: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() { out[pos] = F::cast_from(input[pos]); }
}
