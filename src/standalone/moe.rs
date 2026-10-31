use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, quantization::packed::decode_code};

#[ruda(launch)]
pub fn route<B: Float>(logits: &Array<f32>, bias: &Array<B>, weights: &mut Array<f32>,
    selected: &mut Array<i32>, local_start: i32, factor: f32, #[define(B)] _bias: StorageType,
) {
    let token = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    if token < logits.len() / 256 {
        let lane = UNIT_POS_PLANE as usize;
        let mut probability = Array::<f32>::new(8usize);
        let mut corrected = Array::<f32>::new(8usize);
        let mut ids = Array::<u32>::new(8usize);
        let mut values = Array::<f32>::new(8usize);
        crate::warp_routing::probabilities(logits, bias, &mut probability, &mut corrected, token, 8usize, true, false);
        let denominator = crate::warp_routing::select(&probability, &corrected, &mut ids,
            &mut values, 8usize, 4usize, 8usize, true) + 1e-20f32;
        let global = local_start + lane as i32;
        let mut weight = 0.0f32;
        let mut picked = 0i32;
        #[unroll]
        for slot in 0usize..8 {
            if global == ids[slot] as i32 { weight = values[slot]; picked = 1; }
        }
        weights[token * 32 + lane] = (weight / denominator) * factor;
        selected[token * 32 + lane] = picked;
    }
}

#[ruda(launch)]
pub fn compact(selected: &Array<i32>, rows: &mut Array<u32>, counts: &mut Array<u32>,
    positions: &mut Array<i32>, tokens: u32,
) {
    let expert = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut count = 0u32;
    let mut base = 0usize;
    while base < tokens as usize {
        let token = base + lane;
        let mut take = 0u32;
        if token < tokens as usize { take = selected[token * 32 + expert] as u32; }
        let prefix = plane_exclusive_sum(take);
        let total = plane_sum(take);
        if token < tokens as usize {
            let mut position = -1i32;
            if take != 0 {
                position = (count + prefix) as i32;
                rows[expert * tokens as usize + position as usize] = token as u32;
            }
            positions[token * 32 + expert] = position;
        }
        count += total;
        base += 32;
    }
    if lane == 0 { counts[expert] = count; }
}

#[ruda(launch)]
pub fn offsets(counts: &Array<u32>, out: &mut Array<u32>) {
    let expert = UNIT_POS as usize;
    let padded = ((counts[expert] + 31) / 32) * 32;
    out[expert] = plane_exclusive_sum(padded);
}

#[ruda(launch)]
pub fn project(input: &Array<u8>, activated: &Array<f32>, input_scale: &Array<f32>,
    weights: &Array<u8>, weight_scale: &Array<f32>, rows: &Array<u32>,
    counts: &Array<u32>, offsets: &Array<u32>, output: &mut Array<f32>, tokens: u32,
    #[comptime] first: bool,
) {
    let width = comptime!(if first { 4096usize } else { 7168usize });
    let inner = comptime!(if first { 7168usize } else { 2048usize });
    let thread = UNIT_POS as usize;
    let ty = thread / 16;
    let tx = thread % 16;
    let mut table = SharedMemory::<u32>::new(65usize);
    let mut a = SharedMemory::<f32>::new(32usize * 20);
    let mut b = SharedMemory::<f32>::new(64usize * 20);
    crate::grouped_fp8::tile_table(counts, &mut table, 32usize);
    let m_tiles = table[32usize] as usize;
    let mut task = RUDA_POS_X as usize;
    while task < m_tiles * (width / 64) {
        let m_tile = task % m_tiles;
        let expert = crate::grouped_fp8::tile_expert(&table, m_tile);
        let row_base = (m_tile - table[expert] as usize) * 32;
        let col_base = task / m_tiles * 64;
        let count = table[33usize + expert] as usize;
        let offset = offsets[expert] as usize;
        let mut acc = Array::<f32>::new(16usize);
        #[unroll] for j in 0usize..16 { acc[j] = 0.0; }
        for tile in 0usize..(inner / 16) {
            #[unroll] for j in 0usize..4 {
                let p = thread + j * 128;
                let r = row_base + p / 16;
                let c = tile * 16 + p % 16;
                let mut value = 0.0f32;
                if r < count {
                    if comptime!(first) {
                        let token = rows[expert * tokens as usize + r] as usize;
                        value = decode_code(input[token * inner + c] as u32, 18u32)
                            * input_scale[(c / 128) * tokens as usize + token];
                    } else { value = activated[(offset + r) * inner + c]; }
                }
                a[(p / 16) * 20 + p % 16] = value;
            }
            #[unroll] for j in 0usize..8 {
                let p = thread + j * 128;
                let r = col_base + p / 16;
                let c = tile * 16 + p % 16;
                b[(p / 16) * 20 + p % 16] = decode_code(weights[(expert * width + r) * inner + c] as u32, 18u32)
                    * weight_scale[(expert * (width / 128) + r / 128) * (inner / 128) + c / 128];
            }
            sync_ruda();
            #[unroll] for k in 0usize..16 {
                #[unroll] for i in 0usize..4 {
                    let av = a[(ty * 4 + i) * 20 + k];
                    #[unroll] for j in 0usize..4 {
                        acc[i * 4 + j] = fma(av, b[(tx * 4 + j) * 20 + k], acc[i * 4 + j]);
                    }
                }
            }
            sync_ruda();
        }
        #[unroll] for i in 0usize..4 {
            let r = row_base + ty * 4 + i;
            if r < count {
                #[unroll] for j in 0usize..4 {
                    output[(offset + r) * width + col_base + tx * 4 + j] = acc[i * 4 + j];
                }
            }
        }
        task += RUDA_COUNT_X as usize;
    }
}

#[ruda(launch)]
pub fn activate(input: &Array<f32>, counts: &Array<u32>, offsets: &Array<u32>, out: &mut Array<f32>) {
    let expert = RUDA_POS_Y as usize;
    let p = (RUDA_POS_X * RUDA_DIM_X + UNIT_POS_X) as usize;
    if p < counts[expert] as usize * 2048 {
        let row = offsets[expert] as usize + p / 2048;
        let col = p % 2048;
        let x1 = input[row * 4096 + col];
        let x2 = input[row * 4096 + 2048 + col];
        out[row * 2048 + col] = (x2 / (1.0f32 + (-x2).exp())) * x1;
    }
}

#[ruda(launch)]
pub fn combine<F: Float>(input: &Array<f32>, positions: &Array<i32>, offsets: &Array<u32>,
    weights: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let token = p / 7168;
        let col = p % 7168;
        let mut value = 0.0f32;
        for expert in 0usize..32 {
            let pos = positions[token * 32 + expert];
            if pos >= 0 {
                let row = offsets[expert] as usize + pos as usize;
                let contribution = input[row * 7168 + col] * weights[token * 32 + expert];
                value += contribution;
            }
        }
        out[p] = F::cast_from(value);
    }
}
