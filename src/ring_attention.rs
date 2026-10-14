//! Online-softmax ring blocks with global normalization and FP32 gradient accumulation.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::dense_attention::{keep, keep_global};

#[ruda]
fn visible(qmeta: &Array<i64>, kmeta: &Array<i64>, query: usize, key: usize,
    left: i64, right: i64, #[comptime] causal: bool, #[comptime] indexed: bool) -> bool {
    let stride = comptime!(if indexed { 4usize } else { 3usize });
    let document = qmeta[query * stride];
    let distance = kmeta[key * 3 + 1] - qmeta[query * stride + 1];
    let mut allowed = document >= 0 && document == kmeta[key * 3];
    if comptime!(causal) { allowed = allowed && distance <= 0; }
    if left >= 0 { allowed = allowed && distance >= -left; }
    if right >= 0 { allowed = allowed && distance <= right; }
    allowed
}

#[ruda]
fn dropout_weight(qmeta: &Array<i64>, kmeta: &Array<i64>, query: usize, key: usize,
    head: usize, heads: u32, global_keys: u32, seed_low: u32, seed_high: u32,
    base_low: u32, base_high: u32, p: f32, #[comptime] indexed: bool) -> f32 {
    if comptime!(indexed) {
        let position = qmeta[query * 4 + 2] + i64::cast_from(head) * qmeta[query * 4 + 3] + kmeta[key * 3 + 2];
        keep_global(position, seed_low, seed_high, base_low, base_high, p)
    } else {
        let rng_row = u32::cast_from(qmeta[query * 3 + 2]) * heads + head as u32;
        let blocks = global_keys / 4 + u32::cast_from(global_keys % 4 != 0);
        let low = base_low + rng_row * blocks;
        let mut high = base_high + rng_row.mul_hi(blocks);
        if low < base_low { high += 1; }
        keep(usize::cast_from(kmeta[key * 3 + 2]), seed_low, seed_high, low, high, p)
    }
}

#[ruda(launch)]
pub fn forward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    qmeta: &Array<i64>, kmeta: &Array<i64>, accumulator: &mut Array<f32>, stats: &mut Array<f32>,
    heads: u32, kv_heads: u32, queries: u32, keys: u32, global_keys: u32,
    scale: f32, p: f32, softcap: f32, left: i64, right: i64,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] dropout: bool, #[comptime] capped: bool, #[comptime] indexed: bool,
    #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let token_q = row / heads as usize;
    let head = row % heads as usize;
    let batch = token_q / queries as usize;
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
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        acc[j] = 0.0;
        if d < value_dim { acc[j] = accumulator[row * value_dim + d]; }
    }
    let mut maximum = stats[row * 2];
    let mut denominator = stats[row * 2 + 1];
    for index in 0usize..keys as usize {
        let token_k = batch * keys as usize + index;
        if visible(qmeta, kmeta, token_q, token_k, left, right, causal, indexed) {
            let cache_row = token_k * kv_heads as usize + kh;
            let mut partial = 0.0f32;
            #[unroll]
            for j in 0usize..((key_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < key_dim { partial = fma(query[j], f32::cast_from(k[cache_row * key_dim + d]), partial); }
            }
            let mut score = plane_sum(partial) * scale;
            if comptime!(capped) { score = softcap * (score / softcap).tanh(); }
            let mut next_max = score;
            let mut correction = 0.0f32;
            if denominator != 0.0 {
                next_max = f32::max(maximum, score);
                correction = (maximum - next_max).exp();
            }
            let probability = (score - next_max).exp();
            denominator = fma(denominator, correction, probability);
            maximum = next_max;
            let mut multiplier = 1.0f32;
            if comptime!(dropout) {
                multiplier = dropout_weight(qmeta, kmeta, token_q, token_k, head, heads, global_keys,
                    seed_low, seed_high, base_low, base_high, p, indexed);
            }
            #[unroll]
            for j in 0usize..((value_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < value_dim {
                    acc[j] = fma(probability * multiplier, f32::cast_from(v[cache_row * value_dim + d]), acc[j] * correction);
                }
            }
        }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < value_dim { accumulator[row * value_dim + d] = acc[j]; }
    }
    if lane == 0 { stats[row * 2] = maximum; stats[row * 2 + 1] = denominator; }
}

#[ruda(launch)]
pub fn normalize<F: Float>(accumulator: &mut Array<f32>, stats: &Array<f32>, output: &mut Array<F>,
    width: u32, #[define(F)] _dtype: StorageType) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let denominator = stats[(index / width as usize) * 2 + 1];
        let mut result = 0.0f32;
        if denominator != 0.0 { result = accumulator[index] / denominator; }
        accumulator[index] = result;
        output[index] = F::cast_from(result);
    }
}

#[ruda(launch)]
pub fn backward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    qmeta: &Array<i64>, kmeta: &Array<i64>, grad: &Array<F>, output: &Array<f32>, stats: &Array<f32>,
    dq: &mut Array<f32>, dk: &mut Array<Atomic<f32>>, dv: &mut Array<Atomic<f32>>,
    heads: u32, kv_heads: u32, queries: u32, keys: u32, global_keys: u32,
    scale: f32, p: f32, softcap: f32, left: i64, right: i64,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] dropout: bool, #[comptime] capped: bool, #[comptime] indexed: bool,
    #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let token_q = row / heads as usize;
    let head = row % heads as usize;
    let batch = token_q / queries as usize;
    let kh = head / (heads / kv_heads) as usize;
    let maximum = stats[row * 2];
    let denominator = stats[row * 2 + 1];
    let mut query = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut qgrad = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut dy = Array::<f32>::new((value_dim + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        query[j] = 0.0; qgrad[j] = 0.0;
        if d < key_dim {
            query[j] = f32::cast_from(q[row * key_dim + d]);
            qgrad[j] = dq[row * key_dim + d];
        }
    }
    let mut local_delta = 0.0f32;
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        dy[j] = 0.0;
        if d < value_dim {
            dy[j] = f32::cast_from(grad[row * value_dim + d]);
            local_delta = fma(dy[j], output[row * value_dim + d], local_delta);
        }
    }
    let delta = plane_sum(local_delta);
    for index in 0usize..keys as usize {
        let token_k = batch * keys as usize + index;
        if denominator != 0.0 && visible(qmeta, kmeta, token_q, token_k, left, right, causal, indexed) {
            let cache_row = token_k * kv_heads as usize + kh;
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
            let mut derivative = 1.0f32;
            if comptime!(capped) {
                let t = (score / softcap).tanh();
                score = softcap * t;
                derivative = 1.0 - t * t;
            }
            let probability = (score - maximum).exp() / denominator;
            let mut multiplier = 1.0f32;
            if comptime!(dropout) {
                multiplier = dropout_weight(qmeta, kmeta, token_q, token_k, head, heads, global_keys,
                    seed_low, seed_high, base_low, base_high, p, indexed);
            }
            let ds = probability * (plane_sum(dp_local) * multiplier - delta) * derivative * scale;
            #[unroll]
            for j in 0usize..((key_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < key_dim {
                    qgrad[j] = fma(ds, f32::cast_from(k[cache_row * key_dim + d]), qgrad[j]);
                    dk[cache_row * key_dim + d].fetch_add(ds * query[j]);
                }
            }
            #[unroll]
            for j in 0usize..((value_dim + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < value_dim { dv[cache_row * value_dim + d].fetch_add(probability * multiplier * dy[j]); }
            }
        }
    }
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < key_dim { dq[row * key_dim + d] = qgrad[j]; }
    }
}
