use ruda_core::e4m3;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, quantization::packed::{decode_code, encode_code}};

#[ruda(launch)]
pub fn amax<F: Float>(x: &Array<Vector<F, Const<8>>>, partial: &mut Array<f32>, #[define(F)] _f: StorageType) {
    let t = UNIT_POS as usize;
    let mut i = ABSOLUTE_POS as usize;
    let mut value = 0.0f32;
    while i < x.len() {
        let v = Vector::<f32, Const<8>>::cast_from(x[i]);
        #[unroll]
        for d in 0usize..8 { value = f32::max(value, f32::abs(v[d])); }
        i += RUDA_COUNT_X as usize * 256;
    }
    let mut shared = SharedMemory::<f32>::new(8usize);
    let warp = plane_max(value);
    if t % 32 == 0 { shared[t / 32] = warp; }
    sync_ruda();
    let mut part = 0.0f32;
    if t % 32 < 8 { part = shared[t % 32]; }
    let total = plane_max(part);
    if t == 0 { partial[RUDA_POS_X as usize] = total; }
}

#[ruda(launch)]
pub fn quantize<F: Float>(x: &Array<Vector<F, Const<8>>>, partial: &Array<f32>,
    out: &mut Array<Vector<half::bf16, Const<8>>>, global: &mut Array<f32>,
    #[define(F)] _f: StorageType, #[comptime] round_block_scale: bool,
) {
    let t = UNIT_POS as usize;
    let mut i = t;
    let mut maximum = 0.0f32;
    while i < partial.len() { maximum = f32::max(maximum, partial[i]); i += 128; }
    let mut scratch = SharedMemory::<f32>::new(4usize);
    let warp = plane_max(maximum);
    if t % 32 == 0 { scratch[t / 32] = warp; }
    sync_ruda();
    let mut part = 0.0f32;
    if t % 32 < 4 { part = scratch[t % 32]; }
    let amax = plane_max(part);
    let mut encode = f32::min(2688.0f32 / amax, f32::new(f32::MAX));
    if encode == 0.0f32 { encode = 1.0f32; }
    let decode = 1.0f32 / encode;
    if ABSOLUTE_POS == 0 { global[0] = decode; }
    let block = ABSOLUTE_POS as usize;
    if block < x.len() / 2 {
        let lo = Vector::<f32, Const<8>>::cast_from(x[block * 2]);
        let hi = Vector::<f32, Const<8>>::cast_from(x[block * 2 + 1]);
        let mut mx = 0.0f32;
        #[unroll]
        for d in 0usize..8 { mx = f32::max(mx, f32::max(f32::abs(lo[d]), f32::abs(hi[d]))); }
        let mut block_scale = mx / 6.0f32;
        if comptime!(round_block_scale) { block_scale = f32::cast_from(F::cast_from(block_scale)); }
        let sf = f32::cast_from(e4m3::cast_from(f32::min(block_scale * encode, 448.0f32)));
        let inverse = f32::min(1.0f32 / (sf * decode), f32::new(f32::MAX));
        let mut a = Vector::<half::bf16, Const<8>>::empty();
        let mut b = Vector::<half::bf16, Const<8>>::empty();
        #[unroll]
        for d in 0usize..8 {
            let qa = encode_code(f32::clamp(lo[d] * inverse, -6.0f32, 6.0f32), 14u32, true);
            let qb = encode_code(f32::clamp(hi[d] * inverse, -6.0f32, 6.0f32), 14u32, true);
            a[d] = half::bf16::cast_from(decode_code(qa, 14u32) * sf);
            b[d] = half::bf16::cast_from(decode_code(qb, 14u32) * sf);
        }
        out[block * 2] = a;
        out[block * 2 + 1] = b;
    }
}

#[ruda(launch)]
pub fn dequantize(x: &Array<u8>, scales: &Array<u8>, out: &mut Array<Vector<half::bf16, Const<8>>>) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let scale = f32::cast_from(e4m3::reinterpret(scales[i / 2]));
        let mut result = Vector::<half::bf16, Const<8>>::empty();
        #[unroll]
        for d in 0usize..4 {
            let byte = x[i * 4 + d] as u32;
            result[d * 2] = half::bf16::cast_from(decode_code(byte & 15u32, 14u32) * scale);
            result[d * 2 + 1] = half::bf16::cast_from(decode_code(byte >> 4u32, 14u32) * scale);
        }
        out[i] = result;
    }
}

#[ruda(launch)]
pub fn vision_positions(mask: &Array<u8>, positions: &mut Array<i32>, seq: u32, vision: u32) {
    let batch = RUDA_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = SharedMemory::<u32>::new(8usize);
    let mut running = 0u32;
    let mut base = 0usize;
    while base < seq as usize {
        let token = base + t;
        let mut take = 0u32;
        if token < seq as usize { take = select(mask[batch * seq as usize + token] != 0u8, 1u32, 0u32); }
        let local = plane_exclusive_sum(take);
        let total = plane_sum(take);
        if t % 32 == 0 { scratch[t / 32] = total; }
        sync_ruda();
        let mut before = 0u32;
        let mut count = 0u32;
        #[unroll]
        for w in 0usize..8 {
            let n = scratch[w];
            if w < t / 32 { before += n; }
            count += n;
        }
        if token < seq as usize {
            let rank = running + before + local;
            let mut position = -1i32;
            if take != 0u32 && rank < vision { position = rank as i32; }
            positions[batch * seq as usize + token] = position;
        }
        running += count;
        sync_ruda();
        base += 256;
    }
}

#[ruda(launch)]
pub fn vision_merge(ids: &Array<i64>, positions: &Array<i32>, embedding: &Array<Vector<half::bf16, Const<8>>>,
    vision: &Array<Vector<half::bf16, Const<8>>>, out: &mut Array<Vector<half::bf16, Const<8>>>,
    seq: u32, vision_rows: u32, vectors: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let token = i / vectors as usize;
        let col = i % vectors as usize;
        let position = positions[token];
        if position >= 0 {
            let row = (token / seq as usize) * vision_rows as usize + position as usize;
            out[i] = vision[row * vectors as usize + col];
        } else { out[i] = embedding[ids[token] as usize * vectors as usize + col]; }
    }
}

#[ruda(launch)]
pub fn attention_scores(scores: &Array<half::bf16>, mask: &Array<half::bf16>, out: &mut Array<half::bf16>,
    count: u32, columns: u32, stride: u32, queries: u32, heads: u32, factor: f32,
    #[comptime] masked: bool, #[comptime] causal: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let row = i / columns as usize;
        let col = i % columns as usize;
        let offset = row * stride as usize + col;
        let mut value = f32::cast_from(half::bf16::cast_from(f32::cast_from(scores[offset]) * factor));
        if comptime!(masked) {
            let batch = row / (queries * heads) as usize;
            value += f32::cast_from(mask[(batch * queries as usize + row % queries as usize) * columns as usize + col]);
        }
        if comptime!(causal) {
            if col > row % queries as usize + columns as usize - queries as usize { value = f32::new(f32::NEG_INFINITY); }
        }
        out[offset] = half::bf16::cast_from(value);
    }
}

#[ruda(launch)]
pub fn router6(logits: &Array<f32>, bias: &Array<f32>, indices: &mut Array<i64>, weights: &mut Array<f32>,
    tokens: u32, experts: u32, factor: f32,
) {
    let token = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if token < tokens as usize {
        let mut scores = Array::<f32>::new(16usize);
        #[unroll]
        for j in 0usize..16 {
            let e = j * 32 + lane;
            let mut v = f32::new(f32::NEG_INFINITY);
            if e < experts as usize { v = 1.0f32 / (1.0f32 + (-logits[token * experts as usize + e]).exp()) + bias[e]; }
            scores[j] = v;
        }
        let mut chosen = Array::<f32>::new(6usize);
        let mut ids = Array::<u32>::new(6usize);
        #[unroll]
        for pick in 0usize..6 {
            let mut best = f32::new(f32::NEG_INFINITY);
            let mut id = 0xFFFFFFFFu32;
            #[unroll]
            for j in 0usize..16 {
                if scores[j] > best { best = scores[j]; id = (j * 32 + lane) as u32; }
            }
            let value = plane_max(best);
            let winner = plane_min(select(value == best, id, 0xFFFFFFFFu32));
            chosen[pick] = value;
            ids[pick] = winner;
            #[unroll]
            for j in 0usize..16 {
                if (j * 32 + lane) as u32 == winner { scores[j] = f32::new(f32::NEG_INFINITY); }
            }
        }
        let threshold = chosen[5usize];
        let mut ordered = Array::<f32>::new(6usize);
        let mut ordered_ids = Array::<u32>::new(6usize);
        #[unroll]
        for pick in 0usize..6 {
            let mut rank = 0usize;
            #[unroll]
            for other in 0usize..6 {
                if chosen[other] > threshold {
                    if chosen[pick] == threshold || ids[other] < ids[pick] { rank += 1; }
                } else if chosen[pick] == threshold && ids[other] < ids[pick] { rank += 1; }
            }
            ordered[rank] = chosen[pick];
            ordered_ids[rank] = ids[pick];
        }
        let mut total = 0.0f32;
        #[unroll]
        for pick in 0usize..6 { total += ordered[pick]; }
        if lane == 0 {
            #[unroll]
            for pick in 0usize..6 {
                indices[token * 6 + pick] = ordered_ids[pick] as i64;
                weights[token * 6 + pick] = ordered[pick] / (total + 1e-20f32) * factor;
            }
        }
    }
}
