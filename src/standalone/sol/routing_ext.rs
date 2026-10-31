use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};
use super::common::rnd;

#[ruda(launch)]
pub fn expert_pair_weights(x: &Array<f32>, ids: &Array<i64>, out: &mut Array<f32>, experts: u32, topk: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let e = ids[i] as usize;
        let mut value = 0.0f32;
        if e < experts as usize { value = x[i / topk as usize * experts as usize + e]; }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn topk<F: Float>(scores: &Array<F>, ids: &mut Array<i64>, weights: &mut Array<F>, experts: u32, k: u32,
    renorm: u32, epsilon: f32, factor: f32, #[define(F)] _f: StorageType,
) {
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS_PLANE as usize;
    if row < ids.len() / k as usize {
        let mut values = Array::<f32>::new(16usize);
        #[unroll]
        for j in 0usize..16 {
            let e = lane + j * 32;
            let mut v = f32::NEG_INFINITY;
            if e < experts as usize { v = f32::cast_from(scores[row * experts as usize + e]); }
            values[j] = v;
        }
        let mut chosen = Array::<f32>::new(8usize);
        let mut total = 0.0f32;
        for rank in 0usize..k as usize {
            let mut best = f32::NEG_INFINITY;
            #[unroll]
            for j in 0usize..16 { best = f32::max(best, values[j]); }
            best = plane_max(best);
            let mut index = 0x7fffffffu32;
            #[unroll]
            for j in 0usize..16 {
                if values[j] == best { index = u32::min(index, (lane + j * 32) as u32); }
            }
            index = plane_min(index);
            if lane == 0 { ids[row * k as usize + rank] = i64::cast_from(index); }
            chosen[rank] = best; total += best;
            #[unroll]
            for j in 0usize..16 { if lane + j * 32 == index as usize { values[j] = f32::NEG_INFINITY; } }
        }
        let denominator = rnd::<F>(rnd::<F>(total) + epsilon);
        if lane == 0 {
            for rank in 0usize..k as usize {
                let mut v = chosen[rank];
                if renorm != 0 { v = rnd::<F>(v / denominator); }
                weights[row * k as usize + rank] = F::cast_from(v * factor);
            }
        }
    }
}

#[ruda(launch)]
pub fn ordered_combine<F: Float, W: Float>(y: &Array<F>, inverse: &Array<u32>, ids: &Array<i64>, weights: &Array<W>,
    out: &mut Array<F>, width: u32, topk: u32, #[define(F)] _f: StorageType, #[define(W)] _w: StorageType,
    #[comptime] rounded: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let token = i / width as usize;
        let d = i % width as usize;
        let mut previous = -1i64;
        let mut result = 0.0f32;
        for step in 0u32..topk {
            let mut next = 0x7fffffff_i64;
            let mut selected = 0usize;
            for j in 0usize..topk as usize {
                let e = ids[token * topk as usize + j] * topk as i64 + j as i64;
                if e > previous && e < next { next = e; selected = j; }
            }
            if next != 0x7fffffff_i64 {
                let pair = token * topk as usize + selected;
                let row = inverse[pair] as usize;
                let mut product = f32::cast_from(y[row * width as usize + d]) * f32::cast_from(weights[pair]);
                if rounded { product = rnd::<F>(product); }
                result += product;
                if rounded { result = rnd::<F>(result); }
                previous = next;
            }
        }
        out[i] = F::cast_from(result);
    }
}

#[ruda(launch)]
pub fn scatter_scores(values: &Array<f32>, ids: &Array<i64>, out: &mut Array<f32>, experts: u32, topk: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let token = i / experts as usize;
        let e = i % experts as usize;
        let mut sum = 0.0f32;
        for j in 0usize..topk as usize {
            if ids[token * topk as usize + j] == e as i64 { sum += values[token * topk as usize + j]; }
        }
        out[i] = sum;
    }
}

#[ruda(launch)]
pub fn router_backward(probs: &Array<f32>, ids: &Array<i64>, sums: &Array<f32>, grad_weights: &Array<half::bf16>,
    grad_mask: &Array<half::bf16>, grad_logits: &Array<half::bf16>, out: &mut Array<f32>, experts: u32, topk: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let tokens = out.len() / experts as usize;
        let token = i / experts as usize;
        let expert = i % experts as usize;
        let mut sum = 0.0f32;
        for j in 0usize..topk as usize {
            let pair = token * topk as usize + j;
            sum += f32::cast_from(grad_weights[pair]) * probs[token * experts as usize + ids[pair] as usize] / sums[token];
        }
        let mut value = f32::cast_from(grad_logits[i]);
        for j in 0usize..topk as usize {
            let pair = token * topk as usize + j;
            if ids[pair] == expert as i64 {
                value += f32::cast_from(grad_weights[pair]) / sums[token] - sum / sums[token]
                    + f32::cast_from(grad_mask[(expert * topk as usize + j) * tokens + token]);
            }
        }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn pair_permute(x: &Array<f32>, inverse: &Array<u32>, out: &mut Array<f32>, #[comptime] reverse: bool) {
    let i = ABSOLUTE_POS as usize;
    if i < inverse.len() {
        let row = inverse[i] as usize;
        if reverse { out[i] = x[row]; }
        else { out[row] = x[i]; }
    }
}
