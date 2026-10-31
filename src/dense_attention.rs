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
pub(crate) fn keep(position: usize, seed_low: u32, seed_high: u32, base_low: u32, base_high: u32, p: f32) -> f32 {
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

#[ruda]
pub(crate) fn keep_global(position: i64, seed_low: u32, seed_high: u32,
    base_low: u32, base_high: u32, p: f32) -> f32 {
    let block = position / 4;
    let low = base_low + u32::cast_from(block);
    let mut high = base_high + u32::cast_from(block >> 32);
    if low < base_low { high += 1; }
    keep(usize::cast_from(position % 4), seed_low, seed_high, low, high, p)
}

#[ruda(launch)]
pub fn dropout_mask<F: Float>(output: &mut Array<F>, heads: u32, queries: u32, keys: u32,
    global_heads: i64, global_queries: i64, global_keys: i64, head_start: i64, query_start: i64, key_start: i64,
    p: f32, seed_low: u32, seed_high: u32, base_low: u32, base_high: u32, #[define(F)] _dtype: StorageType) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let key = index % keys as usize;
        let query = (index / keys as usize) % queries as usize;
        let head = (index / keys as usize / queries as usize) % heads as usize;
        let batch = index / keys as usize / queries as usize / heads as usize;
        let position = (((i64::cast_from(batch) * global_heads + head_start + i64::cast_from(head)) * global_queries
            + query_start + i64::cast_from(query)) * global_keys) + key_start + i64::cast_from(key);
        output[index] = F::cast_from(keep_global(position, seed_low, seed_high, base_low, base_high, p));
    }
}

#[ruda(launch)]
pub fn forward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, mask: &Tensor<f32>,
    out: &mut Array<F>, stats: &mut Array<f32>,
    heads: u32, kv_heads: u32, queries: u32, keys: u32, scale: f32, p: f32,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    global_heads: i64, head_start: i64,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] lanes: usize, #[comptime] causal: bool, #[comptime] mask_kind: u32,
    #[comptime] dropout: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize * (RUDA_DIM as usize / lanes) + UNIT_POS as usize / lanes;
    let lane = UNIT_POS as usize % lanes;
    if row >= stats.len() / 2 { return; }
    let head = (row / queries as usize) % heads as usize;
    let batch = row / (heads as usize * queries as usize);
    let rng_row = ((i64::cast_from(batch) * global_heads + head_start + i64::cast_from(head)) * i64::cast_from(queries))
        + i64::cast_from(row % queries as usize);
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
            if denominator != 0.0 { old_scale = softmax_exp(maximum - next_max); }
            weight = softmax_exp(score - next_max);
            denominator = fma(denominator, old_scale, weight);
            maximum = next_max;
        }
        let mut multiplier = 1.0f32;
        if comptime!(dropout) { multiplier = keep_global(rng_row * i64::cast_from(keys) + i64::cast_from(token), seed_low, seed_high, base_low, base_high, p); }
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
    global_heads: i64, head_start: i64,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] mask_kind: u32, #[comptime] dropout: bool,
    #[comptime] need_mask: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize * (RUDA_DIM as usize / lanes) + UNIT_POS as usize / lanes;
    let lane = UNIT_POS as usize % lanes;
    if row >= stats.len() / 2 { return; }
    let head = (row / queries as usize) % heads as usize;
    let batch = row / (heads as usize * queries as usize);
    let rng_row = ((i64::cast_from(batch) * global_heads + head_start + i64::cast_from(head)) * i64::cast_from(queries))
        + i64::cast_from(row % queries as usize);
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
            if allowed && denominator != 0.0 { probability = softmax_exp(score - maximum) / denominator; }
            let mut multiplier = 1.0f32;
            if comptime!(dropout) { multiplier = keep_global(rng_row * i64::cast_from(keys) + i64::cast_from(token), seed_low, seed_high, base_low, base_high, p); }
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
                        let delta = ds * scale * query[j];
                        if delta != 0.0 { dk[cache_row * key_dim + d].fetch_add(delta); }
                    }
                }
                #[unroll]
                for j in 0usize..((value_dim + lanes - 1) / lanes) {
                    let d = lane + j * lanes;
                    let mut dropped_probability = probability * multiplier;
                    if comptime!(dropout) { if p == 1.0 { dropped_probability = 0.0; } }
                    if d < value_dim {
                        let delta = dropped_probability * dy[j];
                        if delta != 0.0 { dv[cache_row * value_dim + d].fetch_add(delta); }
                    }
                }
                if comptime!(need_mask) { if lane == 0 && ds != 0.0 { dm[mi].fetch_add(ds); } }
            }
        }
    }
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < key_dim { dq[row * key_dim + d] = F::cast_from(qgrad[j]); }
    }
}

// Four warps own one query row. Cache only its P and dP in shared memory,
// retaining FP32 probabilities and the unrounded Jacobian contraction.
#[ruda(fast_math = FastMath::ReducedPrecision.into())]
pub(crate) fn flash_exp2(value: f32) -> f32 { f32::powf(2.0f32,value) }

#[ruda]
pub(crate) fn softmax_exp(value:f32)->f32 { flash_exp2(value*core::f32::consts::LOG2_E) }

#[ruda(fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
pub(crate) fn flash_log(value: f32) -> f32 { value.ln() }

// Native Flash-compatible storage boundaries: reverse 64-key blocks, BF16
// unnormalized P, FP32 accumulation, and the rounded O saved for backward.
#[ruda(launch)]
pub fn forward_flash<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    out: &mut Array<F>, stats: &mut Array<f32>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let head = (row / 512) % heads as usize;
    let batch = row / (heads as usize * 512);
    let kh = head / (heads / kv_heads) as usize;
    let mut cache = SharedMemory::<f32>::new(768usize);
    let mut query = Array::<f32>::new(4usize);
    let mut acc = Array::<f32>::new(4usize);
    #[unroll]
    for j in 0usize..4 {
        query[j] = f32::cast_from(q[row * 128 + lane + j * 32]);
        acc[j] = 0.0;
    }
    let scale_log2 = scale * 1.4426950408889634f32;
    let mut maximum = f32::new(f32::NEG_INFINITY);
    let mut denominator = 0.0f32;
    let mut blocks = 8usize;
    if comptime!(causal) { blocks = (row % 512 / 128 + 1) * 2; }
    for step in 0usize..blocks {
        let block = blocks - 1 - step;
        for part in 0usize..16 {
            let local = part * 4 + warp;
            let token = block * 64 + local;
            let kr = (batch * kv_heads as usize + kh) * 512 + token;
            let mut dot = 0.0f32;
            #[unroll]
            for j in 0usize..4 { dot = fma(query[j], f32::cast_from(k[kr * 128 + lane + j * 32]), dot); }
            let mut score = plane_sum(dot);
            if comptime!(causal) { if token > row % 512 { score = f32::NEG_INFINITY; } }
            if lane == 0 { cache[local] = score; }
        }
        sync_ruda();
        let mut candidate = f32::new(f32::NEG_INFINITY);
        if thread < 64 { candidate = cache[thread]; }
        let wm = plane_max(candidate);
        if lane == 0 { cache[192 + warp] = wm; }
        sync_ruda();
        if thread == 0 {
            let mut next = maximum;
            #[unroll]
            for w in 0usize..4 { next = f32::max(next,cache[192 + w]); }
            let mut rescale = 0.0f32;
            if maximum != f32::NEG_INFINITY { rescale = flash_exp2((maximum-next)*scale_log2); }
            cache[200] = rescale;
            cache[201] = next;
        }
        sync_ruda();
        maximum = cache[201];
        let mut max_scaled = 0.0f32;
        if maximum != f32::NEG_INFINITY { max_scaled = maximum * scale_log2; }
        let mut probability = 0.0f32;
        if thread < 64 {
            probability = flash_exp2(cache[thread] * scale_log2 - max_scaled);
            cache[128 + thread] = f32::cast_from(F::cast_from(probability));
        }
        let ws = plane_sum(probability);
        if lane == 0 { cache[196 + warp] = ws; }
        sync_ruda();
        if thread == 0 {
            let mut total = 0.0f32;
            #[unroll]
            for w in 0usize..4 { total += cache[196 + w]; }
            cache[202] = denominator * cache[200] + total;
        }
        sync_ruda();
        denominator = cache[202];
        #[unroll]
        for j in 0usize..4 { acc[j] *= cache[200]; }
        for part in 0usize..16 {
            let local = part * 4 + warp;
            let kr = (batch * kv_heads as usize + kh) * 512 + block * 64 + local;
            let p = cache[128 + local];
            #[unroll]
            for j in 0usize..4 { acc[j] = fma(p,f32::cast_from(v[kr * 128 + lane + j * 32]),acc[j]); }
        }
        sync_ruda();
    }
    #[unroll]
    for j in 0usize..4 { cache[256 + warp * 128 + lane + j * 32] = acc[j]; }
    sync_ruda();
    if warp == 0 {
        #[unroll]
        for j in 0usize..4 {
            let d = lane + j * 32;
            let mut total = 0.0f32;
            #[unroll]
            for w in 0usize..4 { total += cache[256 + w * 128 + d]; }
            let reciprocal = 1.0f32 / denominator;
            let result = F::cast_from(total * reciprocal);
            out[row * 128 + d] = result;
            stats[out.len() / 128 * 2 + row * 128 + d] = f32::cast_from(result);
        }
        if lane == 0 { stats[row * 2] = fma(maximum,scale,flash_log(denominator)); stats[row * 2 + 1] = denominator; }
    }
}

#[ruda(launch)]
pub fn backward_query_cached<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    mask: &Tensor<f32>, grad: &Array<F>, stats: &Array<f32>, dq: &mut Array<F>,
    delta: &mut Array<f32>, heads: u32, kv_heads: u32, scale: f32,
    #[comptime] causal: bool, #[comptime] mask_kind: u32, #[comptime] flash: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let head = (row / 512) % heads as usize;
    let batch = row / (heads as usize * 512);
    let kh = head / (heads / kv_heads) as usize;
    let maximum = stats[row * 2];
    let denominator = stats[row * 2 + 1];
    let mut cache = SharedMemory::<f32>::new(2052usize);
    let mut query = Array::<f32>::new(4usize);
    let mut dy = Array::<f32>::new(4usize);
    let mut qgrad = Array::<f32>::new(4usize);
    #[unroll]
    for j in 0usize..4 {
        query[j] = f32::cast_from(q[row * 128 + lane + j * 32]);
        dy[j] = f32::cast_from(grad[row * 128 + lane + j * 32]);
        qgrad[j] = 0.0;
    }
    let mut expected = 0.0f32;
    for part in 0usize..128 {
        let token = part * 4 + warp;
        let cache_row = (batch * kv_heads as usize + kh) * 512 + token;
        let mut score_local = 0.0f32;
        let mut dp_local = 0.0f32;
        #[unroll]
        for j in 0usize..4 {
            let d = lane + j * 32;
            score_local = fma(query[j], f32::cast_from(k[cache_row * 128 + d]), score_local);
            dp_local = fma(dy[j], f32::cast_from(v[cache_row * 128 + d]), dp_local);
        }
        let raw_score = plane_sum(score_local);
        let mut score = raw_score * scale;
        let dp = plane_sum(dp_local);
        let mut allowed = true;
        if comptime!(causal) { allowed = token <= row % 512; }
        if comptime!(mask_kind != 0) {
            let value = mask[mask_offset(mask, row * 512 + token)];
            if comptime!(mask_kind == 1) { allowed = value != 0.0; }
            else { score += value; }
        }
        if comptime!(mask_kind == 2) { allowed = score != f32::NEG_INFINITY; }
        let mut probability = 0.0f32;
        if allowed && denominator != 0.0 { probability = softmax_exp(score - maximum) / denominator; }
        if comptime!(flash) {
            probability = 0.0;
            if allowed { probability = flash_exp2(raw_score * (scale * 1.4426950408889634f32) - maximum * 1.4426950408889634f32); }
        }
        expected += probability * dp;
        if lane == 0 {
            cache[token] = probability;
            cache[512 + token] = dp;
            cache[1024 + token] = f32::cast_from(allowed);
        }
    }
    if comptime!(flash) {
        expected = 0.0;
        if warp == 0 {
            #[unroll]
            for j in 0usize..4 { expected += dy[j] * stats[dq.len() / 128 * 2 + row * 128 + lane + j * 32]; }
            expected = plane_sum(expected);
        }
    }
    if lane == 0 { cache[2048 + warp] = expected; }
    sync_ruda();
    if warp == 0 {
        let mut value = 0.0f32;
        if lane < 4 { value = cache[2048 + lane]; }
        let total = plane_sum(value);
        if lane == 0 { cache[2048] = total; delta[row] = total; }
    }
    sync_ruda();
    let contraction = cache[2048];
    for part in 0usize..128 {
        let token = part * 4 + warp;
        let cache_row = (batch * kv_heads as usize + kh) * 512 + token;
        let mut ds = 0.0f32;
        if cache[1024 + token] != 0.0 { ds = cache[token] * (cache[512 + token] - contraction); }
        let mut gradient_scale = scale;
        if comptime!(flash) { ds = f32::cast_from(F::cast_from(ds)); gradient_scale = 1.0; }
        #[unroll]
        for j in 0usize..4 {
            qgrad[j] = fma(ds * gradient_scale, f32::cast_from(k[cache_row * 128 + lane + j * 32]), qgrad[j]);
        }
    }
    #[unroll]
    for j in 0usize..4 { cache[1536 + warp * 128 + lane + j * 32] = qgrad[j]; }
    sync_ruda();
    if warp == 0 {
        #[unroll]
        for j in 0usize..4 {
            let d = lane + j * 32;
            let mut total = 0.0f32;
            #[unroll]
            for w in 0usize..4 { total += cache[1536 + w * 128 + d]; }
            if comptime!(flash) { total *= scale; }
            dq[row * 128 + d] = F::cast_from(total);
        }
    }
}

// Each KV row owns its gradient, including every query head in its GQA group.
// No global atomics or full sequence-by-sequence probability tensor is needed.
#[ruda(launch)]
pub fn backward_kv_owned<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    mask: &Tensor<f32>, grad: &Array<F>, stats: &Array<f32>, delta: &Array<f32>,
    dk: &mut Array<F>, dv: &mut Array<F>, heads: u32, kv_heads: u32, scale: f32,
    #[comptime] causal: bool, #[comptime] mask_kind: u32, #[comptime] flash: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let token = row % 512;
    let kh = (row / 512) % kv_heads as usize;
    let batch = row / (kv_heads as usize * 512);
    let group = (heads / kv_heads) as usize;
    let mut key = Array::<f32>::new(4usize);
    let mut value = Array::<f32>::new(4usize);
    let mut key_grad = Array::<f32>::new(4usize);
    let mut value_grad = Array::<f32>::new(4usize);
    let mut sums = SharedMemory::<f32>::new(1024usize);
    let mut grouped = SharedMemory::<f32>::new(256usize);
    if comptime!(flash) {
        grouped[UNIT_POS as usize] = 0.0;
        grouped[128 + UNIT_POS as usize] = 0.0;
        sync_ruda();
    }
    #[unroll]
    for j in 0usize..4 {
        key[j] = f32::cast_from(k[row * 128 + lane + j * 32]);
        value[j] = f32::cast_from(v[row * 128 + lane + j * 32]);
        key_grad[j] = 0.0;
        value_grad[j] = 0.0;
    }
    for part in 0usize..group * 128 {
        let index = part * 4 + warp;
        let query_index = index % 512;
        let head = kh * group + index / 512;
        let qr = (batch * heads as usize + head) * 512 + query_index;
        let mut query = Array::<f32>::new(4usize);
        let mut dy = Array::<f32>::new(4usize);
        let mut score_local = 0.0f32;
        let mut dp_local = 0.0f32;
        #[unroll]
        for j in 0usize..4 {
            query[j] = f32::cast_from(q[qr * 128 + lane + j * 32]);
            dy[j] = f32::cast_from(grad[qr * 128 + lane + j * 32]);
            score_local = fma(query[j], key[j], score_local);
            dp_local = fma(dy[j], value[j], dp_local);
        }
        let raw_score = plane_sum(score_local);
        let mut score = raw_score * scale;
        let dp = plane_sum(dp_local);
        let mut allowed = true;
        if comptime!(causal) { allowed = token <= query_index; }
        if comptime!(mask_kind != 0) {
            let m = mask[mask_offset(mask, qr * 512 + token)];
            if comptime!(mask_kind == 1) { allowed = m != 0.0; }
            else { score += m; }
        }
        if comptime!(mask_kind == 2) { allowed = score != f32::NEG_INFINITY; }
        let mut probability = 0.0f32;
        let mut ds = 0.0f32;
        if allowed {
            if stats[qr * 2 + 1] != 0.0 { probability = softmax_exp(score - stats[qr * 2]) / stats[qr * 2 + 1]; }
            if comptime!(flash) { probability = flash_exp2(raw_score * (scale * 1.4426950408889634f32) - stats[qr * 2] * 1.4426950408889634f32); }
            ds = probability * (dp - delta[qr]);
        }
        let mut gradient_scale = scale;
        if comptime!(flash) {
            ds = f32::cast_from(F::cast_from(ds));
            probability = f32::cast_from(F::cast_from(probability));
            gradient_scale = 1.0;
        }
        #[unroll]
        for j in 0usize..4 {
            key_grad[j] = fma(ds * gradient_scale, query[j], key_grad[j]);
            value_grad[j] = fma(probability, dy[j], value_grad[j]);
        }
        if comptime!(flash) {
            if part % 128 == 127 {
                #[unroll]
                for j in 0usize..4 {
                    let d = lane + j * 32;
                    sums[warp * 128 + d] = key_grad[j]; sums[512 + warp * 128 + d] = value_grad[j];
                }
                sync_ruda();
                if warp == 0 {
                    #[unroll]
                    for j in 0usize..4 {
                        let d = lane + j * 32;
                        let mut total_k = 0.0f32; let mut total_v = 0.0f32;
                        #[unroll]
                        for w in 0usize..4 { total_k += sums[w * 128 + d]; total_v += sums[512 + w * 128 + d]; }
                        grouped[d] += f32::cast_from(F::cast_from(total_k * scale));
                        grouped[128 + d] += f32::cast_from(F::cast_from(total_v));
                    }
                }
                #[unroll]
                for j in 0usize..4 { key_grad[j] = 0.0; value_grad[j] = 0.0; }
                sync_ruda();
            }
        }
    }
    #[unroll]
    for j in 0usize..4 {
        let d = lane + j * 32;
        sums[warp * 128 + d] = key_grad[j];
        sums[512 + warp * 128 + d] = value_grad[j];
    }
    sync_ruda();
    if warp == 0 {
        #[unroll]
        for j in 0usize..4 {
            let d = lane + j * 32;
            let mut total_k = 0.0f32;
            let mut total_v = 0.0f32;
            #[unroll]
            for w in 0usize..4 {
                total_k += sums[w * 128 + d];
                total_v += sums[512 + w * 128 + d];
            }
            if comptime!(flash) { total_k = grouped[d]; total_v = grouped[128 + d]; }
            dk[row * 128 + d] = F::cast_from(total_k);
            dv[row * 128 + d] = F::cast_from(total_v);
        }
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
