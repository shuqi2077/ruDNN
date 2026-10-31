use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};
use super::exp_fast;

#[ruda]
fn logsumexp2(maximum: f32, denominator: f32) -> f32 {
    let mut value = f32::new(f32::NEG_INFINITY);
    if denominator != 0.0 { value = (maximum + denominator.ln()) * core::f32::consts::LOG2_E; }
    // An all-masked nonempty row has NaN values but logsumexp(-inf) = -inf.
    if denominator != denominator {
        value = f32::new(f32::NAN);
        if maximum == f32::new(f32::NEG_INFINITY) { value = f32::new(f32::NEG_INFINITY); }
    }
    value
}

#[ruda(launch)]
pub fn csr_attention<F: Float>(
    q: &Array<F>, k: &Array<F>, v: &Array<F>, qp: &Array<F>, kp: &Array<F>,
    qo: &Array<i32>, kv: &Array<i32>, pages: &Array<i32>, partials: &mut Array<f32>,
    out: &mut Array<F>, lse: &mut Array<f32>,
    scale: f32, splits: u32,
    #[comptime] heads: usize, #[comptime] kv_heads: usize, #[comptime] dim: usize,
    #[comptime] position_dim: usize, #[comptime] prefill: bool, #[comptime] paged: bool,
    #[comptime] masked_nan: bool, #[comptime] warps: usize, #[comptime] finalize: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let split = RUDA_POS_Z as usize;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let mut sequence = row;
    let mut query_begin = row;
    let mut query_end = row + 1;
    if comptime!(prefill) {
        let mut low = 0usize;
        let mut high = qo.len() - 1;
        while low + 1 < high {
            let middle = (low + high) / 2;
            if qo[middle] as usize <= row { low = middle; } else { high = middle; }
        }
        sequence = low;
        query_begin = qo[sequence] as usize;
        query_end = qo[sequence + 1] as usize;
    }
    let start = kv[sequence] as usize;
    let length = (kv[sequence + 1] - kv[sequence]) as usize;
    let mut visible = length;
    if comptime!(prefill) {
        let aligned = row as i32 - query_begin as i32 + 1 + length as i32 - (query_end - query_begin) as i32;
        visible = i32::max(0i32, i32::min(length as i32, aligned)) as usize;
    }
    let base = visible / splits as usize;
    let extra = visible % splits as usize;
    let split_begin = split * base + usize::min(split, extra);
    let split_length = base + usize::cast_from(split < extra);
    let span = split_length / warps;
    let tail = split_length % warps;
    let begin = split_begin + warp * span + usize::min(warp, tail);
    let end = begin + span + usize::cast_from(warp < tail);
    let kv_head = head / (heads / kv_heads);
    let mut query = Array::<f32>::new(dim / 32);
    let mut query_pos = Array::<f32>::new(comptime!((position_dim / 32).max(1)));
    let mut acc = Array::<f32>::new(dim / 32);
    #[unroll] for j in 0usize..(dim / 32) {
        query[j] = f32::cast_from(q[(row * heads + head) * dim + lane + j * 32]);
        acc[j] = 0.0;
    }
    #[unroll] for j in 0usize..(position_dim / 32) {
        query_pos[j] = f32::cast_from(qp[(row * heads + head) * position_dim + lane + j * 32]);
    }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    for token in begin..end {
        let mut cache_token = start + token;
        if comptime!(paged) { cache_token = pages[cache_token] as usize; }
        let cache_row = cache_token * kv_heads + kv_head;
        let mut dot = 0.0f32;
        #[unroll] for j in 0usize..(dim / 32) {
            dot = fma(query[j], f32::cast_from(k[cache_row * dim + lane + j * 32]), dot);
        }
        let mut score = plane_sum(dot);
        if comptime!(position_dim != 0) {
            let mut position_dot = 0.0f32;
            #[unroll] for j in 0usize..(position_dim / 32) {
                position_dot = fma(query_pos[j], f32::cast_from(kp[cache_row * position_dim + lane + j * 32]), position_dot);
            }
            score += plane_sum(position_dot);
        }
        score *= scale;
        let mut next = score;
        let mut old_scale = 0.0f32;
        if token != begin {
            next = f32::max(maximum, score);
            old_scale = exp_fast(maximum - next);
        }
        let weight = exp_fast(score - next);
        denominator = fma(denominator, old_scale, weight);
        #[unroll] for j in 0usize..(dim / 32) {
            acc[j] = fma(weight, f32::cast_from(v[cache_row * dim + lane + j * 32]), acc[j] * old_scale);
        }
        maximum = next;
    }
    let merged = crate::attention_state::merge_warps(&mut acc, maximum, denominator,
        warp, lane, dim, 32usize, warps);
    if warp == 0 {
        maximum = merged.0;
        denominator = merged.1;
        if comptime!(masked_nan) {
            if visible == 0 && length != 0 {
                maximum = f32::new(f32::NEG_INFINITY);
                denominator = f32::new(f32::NAN);
                #[unroll] for j in 0usize..(dim / 32) { acc[j] = f32::new(f32::NAN); }
            }
        }
        if comptime!(finalize) {
            #[unroll] for j in 0usize..(dim / 32) {
                let mut value = 0.0f32;
                if denominator != 0.0 { value = acc[j] / denominator; }
                out[(row * heads + head) * dim + lane + j * 32] = F::cast_from(value);
            }
            if lane == 0 { lse[row * heads + head] = logsumexp2(maximum, denominator); }
        } else {
            let offset = ((row * heads + head) * splits as usize + split) * (dim + 2);
            #[unroll] for j in 0usize..(dim / 32) { partials[offset + lane + j * 32] = acc[j]; }
            if lane == 0 { partials[offset + dim] = maximum; partials[offset + dim + 1] = denominator; }
        }
    }
}

#[ruda(launch)]
pub fn merge_lse<F: Float>(partials: &Array<f32>, out: &mut Array<F>, lse: &mut Array<f32>,
    splits: u32, #[comptime] dim: usize, #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut acc = Array::<f32>::new(dim / 32);
    #[unroll] for j in 0usize..(dim / 32) { acc[j] = 0.0; }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    if splits <= 32 {
        let mut part_maximum = f32::new(f32::NEG_INFINITY);
        let mut part_denominator = 0.0f32;
        let mut has_number: u32 = 0u32;
        if lane < splits as usize {
            let offset = (row * splits as usize + lane) * (dim + 2);
            part_denominator = partials[offset + dim + 1];
            if part_denominator != 0.0 {
                let value = partials[offset + dim];
                if value == value { part_maximum = value; has_number = 1; }
            }
        }
        maximum = plane_max(part_maximum);
        let nonempty = plane_sum(u32::cast_from(part_denominator != 0.0));
        let numeric = plane_sum(has_number);
        if nonempty == 0 { maximum = 0.0; }
        else if numeric == 0 { maximum = f32::new(f32::NAN); }
        let mut factor = 0.0f32;
        let mut den = 0.0f32;
        if part_denominator != 0.0 {
            factor = exp_fast(partials[(row * splits as usize + lane) * (dim + 2) + dim] - maximum);
            den = part_denominator * factor;
        }
        denominator = plane_sum(den);
        for split in 0usize..splits as usize {
            let scale = plane_shuffle(factor, split as u32);
            let active = plane_shuffle(u32::cast_from(part_denominator != 0.0), split as u32);
            if active != 0 {
                let offset = (row * splits as usize + split) * (dim + 2);
                #[unroll] for j in 0usize..(dim / 32) {
                    acc[j] = fma(partials[offset + lane + j * 32], scale, acc[j]);
                }
            }
        }
    } else {
        for split in 0usize..splits as usize {
            let offset = (row * splits as usize + split) * (dim + 2);
            let den = partials[offset + dim + 1];
            if den != 0.0 {
                let part_max = partials[offset + dim];
                let mut next = part_max;
                let mut old_scale = 0.0f32;
                if denominator != 0.0 { next = f32::max(maximum, part_max); old_scale = exp_fast(maximum - next); }
                let scale = exp_fast(part_max - next);
                #[unroll] for j in 0usize..(dim / 32) {
                    acc[j] = fma(partials[offset + lane + j * 32], scale, acc[j] * old_scale);
                }
                denominator = fma(den, scale, denominator * old_scale);
                maximum = next;
            }
        }
    }
    #[unroll] for j in 0usize..(dim / 32) {
        let mut value = 0.0f32;
        if denominator != 0.0 { value = acc[j] / denominator; }
        out[row * dim + lane + j * 32] = F::cast_from(value);
    }
    if lane == 0 { lse[row] = logsumexp2(maximum, denominator); }
}
