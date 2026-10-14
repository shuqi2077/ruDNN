use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::dense_attention::keep;

#[ruda]
fn document(schedule: &Array<i64>, token: usize, documents: u32) -> usize {
    let mut low = 0usize;
    let mut high = documents as usize;
    while low < high {
        let middle = (low + high) / 2;
        if i64::cast_from(token) >= schedule[middle * 6 + 1] { low = middle + 1; }
        else { high = middle; }
    }
    low * 6
}

#[ruda]
fn visible(token: usize, query: i64, left: i64, right: i64, #[comptime] causal: bool) -> bool {
    let distance = i64::cast_from(token) - query;
    let mut allowed = true;
    if comptime!(causal) { allowed = distance <= 0; }
    if left >= 0 { allowed = allowed && distance >= -left; }
    if right >= 0 { allowed = allowed && distance <= right; }
    allowed
}

#[ruda(launch)]
pub fn forward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, schedule: &Array<i64>,
    out: &mut Array<F>, stats: &mut Array<f32>, heads: u32, kv_heads: u32, documents: u32,
    scale: f32, p: f32, softcap: f32, left: i64, right: i64,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] lower_right: bool, #[comptime] dropout: bool,
    #[comptime] capped: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let head = row % heads as usize;
    let token_q = row / heads as usize;
    let record = document(schedule, token_q, documents);
    let start_q = usize::cast_from(schedule[record]);
    let length_q = usize::cast_from(schedule[record + 1]) - start_q;
    let start_k = usize::cast_from(schedule[record + 2]);
    let length_k = usize::cast_from(schedule[record + 3]) - start_k;
    let kh = head / (heads / kv_heads) as usize;
    let mut position = i64::cast_from(token_q - start_q);
    if comptime!(lower_right) { position += i64::cast_from(length_k) - i64::cast_from(length_q); }
    let rng_row = u32::cast_from((token_q - start_q) * heads as usize + head);
    let blocks = u32::cast_from((length_k + 3) / 4);
    let offset_low = rng_row * blocks;
    let mut rng_low = base_low + u32::cast_from(schedule[record + 4]);
    let mut rng_high = base_high + u32::cast_from(schedule[record + 5]);
    if rng_low < base_low { rng_high += 1; }
    let previous = rng_low;
    rng_low += offset_low;
    rng_high += rng_row.mul_hi(blocks);
    if rng_low < previous { rng_high += 1; }
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
    for token in 0usize..length_k {
        let cache_row = (start_k + token) * kv_heads as usize + kh;
        let mut partial = 0.0f32;
        #[unroll]
        for j in 0usize..((key_dim + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < key_dim { partial = fma(query[j], f32::cast_from(k[cache_row * key_dim + d]), partial); }
        }
        let mut score = plane_sum(partial) * scale;
        if comptime!(capped) { score = softcap * (score / softcap).tanh(); }
        let mut weight = 0.0f32;
        let mut old_scale = 1.0f32;
        if visible(token, position, left, right, causal) {
            let mut next_max = score;
            if denominator != 0.0 { next_max = f32::max(maximum, score); }
            old_scale = 0.0;
            if denominator != 0.0 { old_scale = (maximum - next_max).exp(); }
            weight = (score - next_max).exp();
            denominator = fma(denominator, old_scale, weight);
            maximum = next_max;
        }
        let mut multiplier = 1.0f32;
        if comptime!(dropout) { multiplier = keep(token, seed_low, seed_high, rng_low, rng_high, p); }
        #[unroll]
        for j in 0usize..((value_dim + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < value_dim {
                acc[j] = fma(weight * multiplier, f32::cast_from(v[cache_row * value_dim + d]), acc[j] * old_scale);
            }
        }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < value_dim {
            let mut result = acc[j];
            if denominator != 0.0 { result /= denominator; }
            out[row * value_dim + d] = F::cast_from(result);
        }
    }
    if lane == 0 { stats[row * 2] = maximum; stats[row * 2 + 1] = denominator; }
}

#[ruda(launch)]
pub fn backward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, schedule: &Array<i64>,
    grad: &Array<F>, stats: &Array<f32>, dq: &mut Array<F>, dk: &mut Array<Atomic<f32>>,
    dv: &mut Array<Atomic<f32>>, heads: u32, kv_heads: u32, documents: u32,
    scale: f32, p: f32, softcap: f32, left: i64, right: i64,
    seed_low: u32, seed_high: u32, base_low: u32, base_high: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] lower_right: bool, #[comptime] dropout: bool,
    #[comptime] capped: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let head = row % heads as usize;
    let token_q = row / heads as usize;
    let record = document(schedule, token_q, documents);
    let start_q = usize::cast_from(schedule[record]);
    let length_q = usize::cast_from(schedule[record + 1]) - start_q;
    let start_k = usize::cast_from(schedule[record + 2]);
    let length_k = usize::cast_from(schedule[record + 3]) - start_k;
    let kh = head / (heads / kv_heads) as usize;
    let mut position = i64::cast_from(token_q - start_q);
    if comptime!(lower_right) { position += i64::cast_from(length_k) - i64::cast_from(length_q); }
    let rng_row = u32::cast_from((token_q - start_q) * heads as usize + head);
    let blocks = u32::cast_from((length_k + 3) / 4);
    let offset_low = rng_row * blocks;
    let mut rng_low = base_low + u32::cast_from(schedule[record + 4]);
    let mut rng_high = base_high + u32::cast_from(schedule[record + 5]);
    if rng_low < base_low { rng_high += 1; }
    let previous = rng_low;
    rng_low += offset_low;
    rng_high += rng_row.mul_hi(blocks);
    if rng_low < previous { rng_high += 1; }
    let maximum = stats[row * 2];
    let denominator = stats[row * 2 + 1];
    let mut query = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut qgrad = Array::<f32>::new((key_dim + lanes - 1) / lanes);
    let mut dy = Array::<f32>::new((value_dim + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        query[j] = 0.0; qgrad[j] = 0.0;
        if d < key_dim { query[j] = f32::cast_from(q[row * key_dim + d]); }
    }
    #[unroll]
    for j in 0usize..((value_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        dy[j] = 0.0;
        if d < value_dim { dy[j] = f32::cast_from(grad[row * value_dim + d]); }
    }
    let mut expected = 0.0f32;
    for pass in 0usize..2 {
        for token in 0usize..length_k {
            let cache_row = (start_k + token) * kv_heads as usize + kh;
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
            let allowed = visible(token, position, left, right, causal);
            let mut probability = 0.0f32;
            if allowed && denominator != 0.0 { probability = (score - maximum).exp() / denominator; }
            let mut multiplier = 1.0f32;
            if comptime!(dropout) { multiplier = keep(token, seed_low, seed_high, rng_low, rng_high, p); }
            let dp = plane_sum(dp_local) * multiplier;
            if pass == 0 { expected += probability * dp; }
            else {
                let mut ds = 0.0f32;
                if allowed { ds = probability * (dp - expected) * derivative * scale; }
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
    }
    #[unroll]
    for j in 0usize..((key_dim + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < key_dim { dq[row * key_dim + d] = F::cast_from(qgrad[j]); }
    }
}
