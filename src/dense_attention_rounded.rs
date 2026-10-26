use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub fn decode<F: Float>(
    q: &Array<F>, k: &Array<F>, v: &Array<F>, output: &mut Array<F>,
    heads: u32, kv_heads: u32, keys: u32, capacity: u32, scale: f32,
    #[comptime] dimension: usize, #[comptime] score_capacity: usize,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let lane = unit % 32;
    let warp = unit / 32;
    let component = unit % 8;
    let group = unit / 8;
    let head = row % heads as usize;
    let batch = row / heads as usize;
    let kv_head = head / (heads / kv_heads) as usize;
    let start = (batch * kv_heads as usize + kv_head) * capacity as usize * dimension;
    let mut scores = SharedMemory::<f32>::new(score_capacity);
    let mut reductions = SharedMemory::<f32>::new(8usize);
    let mut partials = SharedMemory::<f32>::new(8 * dimension);
    let mut query = Array::<f32>::new((dimension + 7) / 8);
    #[unroll]
    for j in 0usize..((dimension + 7) / 8) {
        query[j] = 0.0;
        if component + j * 8 < dimension {
            query[j] = f32::cast_from(q[row * dimension + component + j * 8]);
        }
    }
    let mut base = 0usize;
    while base < keys as usize {
        let token = base + group;
        let mut dot = 0.0f32;
        #[unroll]
        for j in 0usize..((dimension + 7) / 8) {
            if token < keys as usize && component + j * 8 < dimension {
                dot += query[j] * f32::cast_from(k[start + token * dimension + component + j * 8]);
            }
        }
        dot += plane_shuffle_xor(dot, 4u32);
        dot += plane_shuffle_xor(dot, 2u32);
        dot += plane_shuffle_xor(dot, 1u32);
        if component == 0 && token < keys as usize {
            let rounded = f32::cast_from(F::cast_from(dot));
            scores[token] = f32::cast_from(F::cast_from(rounded * scale));
        }
        base += 32;
    }
    sync_ruda();
    let mut maximum = f32::cast_from(f32::NEG_INFINITY);
    let mut token = unit;
    while token < keys as usize {
        maximum = f32::max(maximum, scores[token]);
        token += 256;
    }
    maximum = plane_max(maximum);
    if lane == 0 { reductions[warp] = maximum; }
    sync_ruda();
    maximum = f32::cast_from(f32::NEG_INFINITY);
    #[unroll]
    for i in 0usize..8 { maximum = f32::max(maximum, reductions[i]); }
    sync_ruda();
    let mut denominator = 0.0f32;
    let mut token = unit;
    while token < keys as usize {
        let probability = (scores[token] - maximum).exp();
        scores[token] = probability;
        denominator += probability;
        token += 256;
    }
    denominator = plane_sum(denominator);
    if lane == 0 { reductions[warp] = denominator; }
    sync_ruda();
    denominator = 0.0;
    #[unroll]
    for i in 0usize..8 { denominator += reductions[i]; }
    let mut token = unit;
    while token < keys as usize {
        scores[token] = f32::cast_from(F::cast_from(scores[token] / denominator));
        token += 256;
    }
    sync_ruda();
    let mut sums = Array::<f32>::new((dimension + 31) / 32);
    #[unroll]
    for j in 0usize..((dimension + 31) / 32) { sums[j] = 0.0; }
    let mut token = warp;
    while token < keys as usize {
        let probability = scores[token];
        #[unroll]
        for j in 0usize..((dimension + 31) / 32) {
            let d = lane + j * 32;
            if d < dimension {
                sums[j] += probability * f32::cast_from(v[start + token * dimension + d]);
            }
        }
        token += 8;
    }
    #[unroll]
    for j in 0usize..((dimension + 31) / 32) {
        let d = lane + j * 32;
        if d < dimension { partials[warp * dimension + d] = sums[j]; }
    }
    sync_ruda();
    let mut d = unit;
    while d < dimension {
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0usize..8 { sum += partials[i * dimension + d]; }
        output[row * dimension + d] = F::cast_from(sum);
        d += 256;
    }
}
