//! Recomputed 16x64 Flash backward tiles; no global LxS score/probability buffers.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, ir::MatrixIdent, prelude::*};
use crate::dense_attention::flash_exp2;

#[ruda]
pub(super) fn stage<F: Float>(input: &Array<F>, tile: &mut SharedMemory<F>, row: usize,
    #[comptime] rows: usize, #[comptime] vectorized: bool) {
    let thread = UNIT_POS as usize;
    if comptime!(vectorized) {
        let source = input.to_slice().into_vectorized().with_vector_size::<Const<8>>();
        let mut target = tile.to_slice_mut().into_vectorized().with_vector_size::<Const<8>>();
        for block in 0usize..rows / 8 {
            let r = block * 8 + thread / 16;
            target[r * 17 + thread % 16] = source[(row + r) * 16 + thread % 16];
        }
    } else {
        for block in 0usize..rows / 4 {
            #[unroll]
            for i in 0usize..4 {
                tile[(block * 4 + i) * 136 + thread] = input[(row + block * 4 + i) * 128 + thread];
            }
        }
    }
}

#[ruda]
fn dot_tile<F: Float>(a: &SharedMemory<F>, b: &SharedMemory<F>, tile: &mut SharedMemory<f32>, offset: usize) {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..2 {
        let mut acc = Array::<Vector<f32, Const<2>>>::new(2usize);
        #[unroll]
        for i in 0usize..2 { acc[i] = Vector::new(0.0f32); }
        #[unroll]
        for chunk in 0usize..8 {
            let a_start = (lane % 16) * 136 + chunk * 16 + (lane / 16) * 8;
            let b_start = (warp * 16 + col * 8 + lane % 8) * 136 + chunk * 16 + (lane / 8 % 2) * 8;
            let ar = def.load_matrix::<_, Const<2>>(&a.slice(a_start, a_start + 8), MatrixIdent::A, 4usize, false);
            let br = def.load_matrix::<_, Const<2>>(&b.slice(b_start, b_start + 8), MatrixIdent::B, 2usize, false);
            let value = def.execute(&ar, &br, &acc);
            #[unroll]
            for i in 0usize..2 { acc[i] = value[i]; }
        }
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                tile[offset + r as usize * 64 + warp * 16 + col * 8 + c as usize] = acc[i][j];
            }
        }
    }
}

#[ruda]
fn contract_tile<F: Float>(weights: &SharedMemory<F>, input: &SharedMemory<F>,
    acc: &mut Array<Vector<f32, Const<2>>>) {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        let mut cv = Array::<Vector<f32, Const<2>>>::new(2usize);
        #[unroll]
        for i in 0usize..2 { cv[i] = acc[col * 2 + i]; }
        #[unroll]
        for chunk in 0usize..4 {
            let a_start = (lane % 16) * 64 + chunk * 16 + (lane / 16) * 8;
            let b_start = (chunk * 16 + lane % 16) * 136 + warp * 32 + col * 8;
            let ar = def.load_matrix::<_, Const<2>>(&weights.slice(a_start, a_start + 8), MatrixIdent::A, 4usize, false);
            let br = def.load_matrix::<_, Const<2>>(&input.slice(b_start, b_start + 8), MatrixIdent::B, 2usize, true);
            let value = def.execute(&ar, &br, &cv);
            #[unroll]
            for i in 0usize..2 { cv[i] = value[i]; }
        }
        #[unroll]
        for i in 0usize..2 { acc[col * 2 + i] = cv[i]; }
    }
}

#[ruda(launch_unchecked)]
pub fn contraction<F: Float>(grad: &Array<F>, stats: &Array<f32>, delta: &mut Array<f32>,
    #[define(F)] _dtype: StorageType) {
    let lane = UNIT_POS as usize % 32;
    let row = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    let mut sum = 0.0f32;
    #[unroll]
    for i in 0usize..4 {
        let at = row * 128 + lane + i * 32;
        sum += f32::cast_from(grad[at]) * stats[delta.len() * 2 + at];
    }
    sum = plane_sum(sum);
    if lane == 0 { delta[row] = sum; }
}

#[ruda(launch_unchecked)]
pub fn query_vectorized<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dq: &mut Array<F>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize * 16;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let head = row / 512 % heads as usize;
    let batch = row / (512 * heads as usize);
    let key = (batch * kv_heads as usize + head / (heads / kv_heads) as usize) * 512;
    let mut scores = SharedMemory::<f32>::new(2048usize);
    let mut ds = SharedMemory::<F>::new_aligned(1024usize, 16usize);
    let mut q_tile = SharedMemory::<F>::new_aligned(16usize * 136, 16usize);
    let mut g_tile = SharedMemory::<F>::new_aligned(16usize * 136, 16usize);
    let mut k_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    let mut v_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    stage(q, &mut q_tile, row, 16usize, vectorized);
    stage(grad, &mut g_tile, row, 16usize, vectorized);
    let mut acc = Array::<Vector<f32, Const<2>>>::new(8usize);
    #[unroll]
    for i in 0usize..8 { acc[i] = Vector::new(0.0f32); }
    for block in 0usize..8 {
        stage(k, &mut k_tile, key + block * 64, 64usize, vectorized);
        stage(v, &mut v_tile, key + block * 64, 64usize, vectorized);
        sync_ruda();
        dot_tile(&q_tile, &k_tile, &mut scores, 0usize);
        dot_tile(&g_tile, &v_tile, &mut scores, 1024usize);
        sync_ruda();
        #[unroll]
        for i in 0usize..8 {
            let at = thread + i * 128;
            let qr = row + at / 64;
            let mut value = 0.0f32;
            let mut allowed = true;
            if comptime!(causal) { allowed = block * 64 + at % 64 <= qr % 512; }
            if allowed {
                let p = flash_exp2(scores[at] * (scale * 1.4426950408889634f32) - stats[qr * 2] * 1.4426950408889634f32);
                value = p * (scores[1024 + at] - delta[qr]);
            }
            ds[at] = F::cast_from(value);
        }
        sync_ruda();
        contract_tile(&ds, &k_tile, &mut acc);
        sync_ruda();
    }
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                dq[(row + r as usize) * 128 + warp * 32 + col * 8 + c as usize] = F::cast_from(acc[col * 2 + i][j] * scale);
            }
        }
    }
}

#[ruda(launch_unchecked)]
pub fn key_value_vectorized<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dk: &mut Array<F>, dv: &mut Array<F>,
    heads: u32, kv_heads: u32, scale: f32, #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    key_value_impl(q, k, v, grad, stats, delta, dk, dv, heads, kv_heads, scale, causal, false, vectorized);
}

#[ruda(launch_unchecked)]
pub fn key_value_parallel_vectorized<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dk: &mut Array<F>, dv: &mut Array<F>,
    heads: u32, kv_heads: u32, scale: f32, #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    key_value_impl(q, k, v, grad, stats, delta, dk, dv, heads, kv_heads, scale, causal, true, vectorized);
}

#[ruda]
fn key_value_impl<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dk: &mut Array<F>, dv: &mut Array<F>,
    heads: u32, kv_heads: u32, scale: f32, #[comptime] causal: bool, #[comptime] split_heads: bool, #[comptime] vectorized: bool) {
    let key = RUDA_POS_X as usize * 16;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let kh = key / 512 % kv_heads as usize;
    let batch = key / (512 * kv_heads as usize);
    let group = (heads / kv_heads) as usize;
    let mut scores = SharedMemory::<f32>::new(2048usize);
    let mut ds = SharedMemory::<F>::new_aligned(1024usize, 16usize);
    let mut probability = SharedMemory::<F>::new_aligned(1024usize, 16usize);
    let mut k_tile = SharedMemory::<F>::new_aligned(16usize * 136, 16usize);
    let mut v_tile = SharedMemory::<F>::new_aligned(16usize * 136, 16usize);
    let mut q_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    let mut g_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    stage(k, &mut k_tile, key, 16usize, vectorized);
    stage(v, &mut v_tile, key, 16usize, vectorized);
    let mut sum_k = Array::<Vector<f32, Const<2>>>::new(8usize);
    let mut sum_v = Array::<Vector<f32, Const<2>>>::new(8usize);
    #[unroll]
    for i in 0usize..8 { sum_k[i] = Vector::new(0.0f32); sum_v[i] = Vector::new(0.0f32); }
    let mut count = group;
    if comptime!(split_heads) { count = 1; }
    for index in 0usize..count {
        let mut head_in_group = index;
        if comptime!(split_heads) { head_in_group = RUDA_POS_Y as usize; }
        let query = (batch * heads as usize + kh * group + head_in_group) * 512;
        let mut acc_k = Array::<Vector<f32, Const<2>>>::new(8usize);
        let mut acc_v = Array::<Vector<f32, Const<2>>>::new(8usize);
        #[unroll]
        for i in 0usize..8 { acc_k[i] = Vector::new(0.0f32); acc_v[i] = Vector::new(0.0f32); }
        for block in 0usize..8 {
            stage(q, &mut q_tile, query + block * 64, 64usize, vectorized);
            stage(grad, &mut g_tile, query + block * 64, 64usize, vectorized);
            sync_ruda();
            dot_tile(&k_tile, &q_tile, &mut scores, 0usize);
            dot_tile(&v_tile, &g_tile, &mut scores, 1024usize);
            sync_ruda();
            #[unroll]
            for i in 0usize..8 {
                let at = thread + i * 128;
                let qr = query + block * 64 + at % 64;
                let mut p = 0.0f32;
                let mut value = 0.0f32;
                let mut allowed = true;
                if comptime!(causal) { allowed = key % 512 + at / 64 <= qr % 512; }
                if allowed {
                    p = flash_exp2(scores[at] * (scale * 1.4426950408889634f32) - stats[qr * 2] * 1.4426950408889634f32);
                    value = p * (scores[1024 + at] - delta[qr]);
                }
                ds[at] = F::cast_from(value);
                probability[at] = F::cast_from(p);
            }
            sync_ruda();
            contract_tile(&ds, &q_tile, &mut acc_k);
            contract_tile(&probability, &g_tile, &mut acc_v);
            sync_ruda();
        }
        #[unroll]
        for i in 0usize..8 {
            let mut total_k = sum_k[i];
            let mut total_v = sum_v[i];
            #[unroll]
            for j in 0usize..2 {
                total_k[j] += f32::cast_from(F::cast_from(acc_k[i][j] * scale));
                total_v[j] += f32::cast_from(F::cast_from(acc_v[i][j]));
            }
            sum_k[i] = total_k;
            sum_v[i] = total_v;
        }
    }
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                let mut at = (key + r as usize) * 128 + warp * 32 + col * 8 + c as usize;
                if comptime!(split_heads) { at += RUDA_POS_Y as usize * (dk.len() / group); }
                dk[at] = F::cast_from(sum_k[col * 2 + i][j]);
                dv[at] = F::cast_from(sum_v[col * 2 + i][j]);
            }
        }
    }
}

#[ruda(launch_unchecked)]
pub fn reduce_heads<F: Float>(partial_k: &Array<F>, partial_v: &Array<F>,
    dk: &mut Array<F>, dv: &mut Array<F>, #[comptime] group: usize, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < dk.len() {
        let mut sum_k = 0.0f32;
        let mut sum_v = 0.0f32;
        // Per-head BF16 rounding precedes this ordered FP32 sum, as in the serial kernel.
        #[unroll]
        for head in 0usize..group {
            sum_k += f32::cast_from(partial_k[head * dk.len() + pos]);
            sum_v += f32::cast_from(partial_v[head * dv.len() + pos]);
        }
        dk[pos] = F::cast_from(sum_k);
        dv[pos] = F::cast_from(sum_v);
    }
}
