use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};
use super::common::rnd;

#[ruda(launch)]
pub fn gather_scalar(x: &Array<f32>, indices: &Array<i64>, out: &mut Array<f32>, width: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() { out[i] = x[usize::cast_from(indices[i / width as usize]) * width as usize + i % width as usize]; }
}

#[ruda(launch)]
pub fn position_rope_table(positions: &Array<i64>, cos: &mut Array<half::bf16>, sin: &mut Array<half::bf16>, dim: u32, theta: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < cos.len() {
        let d = i % dim as usize % (dim / 2) as usize;
        let angle = f32::cast_from(positions[i / dim as usize]) / theta.powf((2 * d) as f32 / dim as f32);
        cos[i] = half::bf16::cast_from(angle.cos()); sin[i] = half::bf16::cast_from(angle.sin());
    }
}

#[ruda(launch)]
pub fn relative_table(x: &Array<f32>, out: &mut Array<f32>, size: u32, dim: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let d = i % dim as usize;
        let k = i / dim as usize % size as usize;
        let q = i / dim as usize / size as usize;
        let old = x.len() / dim as usize;
        let pos = q + size as usize - 1 - k;
        let mut source = (pos as f32 + 0.5f32) * old as f32 / (2 * size - 1) as f32 - 0.5f32;
        if source < 0.0f32 { source = 0.0f32; }
        let lo = source as usize;
        let mut hi = lo + 1;
        if hi >= old { hi = old - 1; }
        let fraction = source - lo as f32;
        out[i] = x[lo * dim as usize + d] * (1.0f32 - fraction) + x[hi * dim as usize + d] * fraction;
    }
}

#[ruda(launch)]
pub fn relative_table_grad(x: &Array<f32>, out: &mut Array<f32>, size: u32, dim: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let d = i % dim as usize;
        let old = out.len() / dim as usize;
        let target = i / dim as usize;
        let resized = (2 * size - 1) as usize;
        let mut source = (target as f32 + 0.5f32) * resized as f32 / old as f32 - 0.5f32;
        if source < 0.0f32 { source = 0.0f32; }
        let lo = source as usize;
        let mut hi = lo + 1;
        if hi >= resized { hi = resized - 1; }
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for q in 0usize..size as usize {
            let ka = q as i32 + size as i32 - 1 - lo as i32;
            let kb = q as i32 + size as i32 - 1 - hi as i32;
            if ka >= 0 && ka < size as i32 { a += x[(q * size as usize + ka as usize) * dim as usize + d]; }
            if kb >= 0 && kb < size as i32 { b += x[(q * size as usize + kb as usize) * dim as usize + d]; }
        }
        let f = source - lo as f32;
        out[i] = (a * (1.0f32 - f) + b * f) * (old as f32 / resized as f32);
    }
}

#[ruda(launch)]
pub fn mask_key(x: &Array<f32>, mask: &Array<f32>, out: &mut Array<f32>, heads: u32, queries: u32, keys: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let batch = i / (heads * queries * keys) as usize;
        let key = i % keys as usize;
        let mut v = 0.0f32;
        if mask[batch * keys as usize + key] != 0.0f32 { v = x[i]; }
        out[i] = v;
    }
}

#[ruda(launch)]
pub fn rope_from_frequency(inv: &Array<f32>, cos: &mut Array<f32>, sin: &mut Array<f32>, dim: u32,
    #[comptime] interleaved: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < cos.len() {
        let mut f = i % dim as usize % (dim / 2) as usize;
        if interleaved { f = i % dim as usize / 2; }
        let angle = (i / dim as usize) as f32 * inv[f];
        cos[i] = angle.cos(); sin[i] = angle.sin();
    }
}

#[ruda(launch)]
pub fn multimodal_rope(x: &Array<half::bf16>, positions: &Array<i64>, out: &mut Array<half::bf16>,
    heads: u32, #[comptime] multimodal: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let d = i % 128;
        let token = i / 128 / heads as usize;
        let mut size = 128usize;
        let mut start = 0usize;
        let mut axis = 0usize;
        if multimodal {
            size = 42;
            if d >= 84 { start = 84; size = 44; axis = 2; }
            else if d >= 42 { start = 42; axis = 1; }
        }
        let local = d - start;
        let half = size / 2;
        let mut p = token;
        if multimodal { p = token * 3 + axis; }
        let angle = f32::cast_from(positions[p]) / f32::powf(10000.0f32, (local % half * 2) as f32 / size as f32);
        let c = f32::cast_from(half::bf16::cast_from(angle.cos()));
        let s = f32::cast_from(half::bf16::cast_from(angle.sin()));
        let other = f32::cast_from(x[i - d + start + (local + half) % size]);
        let mut rotated = other;
        if local < half { rotated = -other; }
        let a = f32::cast_from(half::bf16::cast_from(f32::cast_from(x[i]) * c));
        let b = f32::cast_from(half::bf16::cast_from(rotated * s));
        out[i] = half::bf16::cast_from(a + b);
    }
}

#[ruda(launch)]
pub fn sequence_rope<F: Float>(cos: &mut Array<F>, sin: &mut Array<F>, seq: u32, dim: u32, theta: f32,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < cos.len() {
        let col = i % dim as usize;
        let position = i / dim as usize % seq as usize;
        let inv = 1.0f32 / f32::powf(theta, (2 * (col % (dim as usize / 2))) as f32 / dim as f32);
        let phase = position as f32 * inv;
        cos[i] = F::cast_from(phase.cos()); sin[i] = F::cast_from(phase.sin());
    }
}

#[ruda(launch)]
pub fn norm_arithmetic<F: Float>(x: &Array<Vector<F, Const<8>>>, weight: &Array<Vector<F, Const<8>>>,
    bias: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<F, Const<8>>>, width: u32, eps: f32,
    #[define(F)] _f: StorageType,
) {
    let t = UNIT_POS as usize;
    let columns = width as usize / 8;
    let base = RUDA_POS_X as usize * columns;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut total = 0.0f32;
    let mut col = t;
    while col < columns {
        let v = Vector::<f32, Const<8>>::cast_from(x[base + col]);
        #[unroll]
        for d in 0usize..8 { total += v[d]; }
        col += 256;
    }
    let mean = crate::standalone::block_sum(total, &mut scratch, 256usize) / width as f32;
    let mut squares = 0.0f32;
    let mut c = t;
    while c < columns {
        let v = Vector::<f32, Const<8>>::cast_from(x[base + c]);
        #[unroll]
        for d in 0usize..8 { let z = v[d] - mean; squares += z * z; }
        c += 256;
    }
    let variance = crate::standalone::block_sum(squares, &mut scratch, 256usize) / width as f32;
    let mu = rnd::<F>(mean);
    let denominator = rnd::<F>(rnd::<F>(rnd::<F>(variance) + eps).sqrt());
    let mut c2 = t;
    while c2 < columns {
        let v = Vector::<f32, Const<8>>::cast_from(x[base + c2]);
        let w = Vector::<f32, Const<8>>::cast_from(weight[c2]);
        let b = Vector::<f32, Const<8>>::cast_from(bias[c2]);
        let mut result = Vector::<F, Const<8>>::empty();
        #[unroll]
        for d in 0usize..8 {
            let centered = rnd::<F>(v[d] - mu);
            let normalized = rnd::<F>(centered / denominator);
            result[d] = F::cast_from(rnd::<F>(normalized * w[d]) + b[d]);
        }
        out[base + c2] = result;
        c2 += 256;
    }
}

#[ruda(launch)]
pub fn grid_embedding(table: &Array<f32>, inv: &Array<f32>, out: &mut Array<f32>, cos: &mut Array<f32>, sin: &mut Array<f32>,
    frames: u32, height: u32, width: u32, channels: u32, offset: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let hidden = channels as usize;
    if i < (frames * height * width) as usize * hidden {
        let token = i / hidden;
        let col = i % hidden;
        let spatial = token % (height * width) as usize;
        let tile = spatial / 4;
        let y = tile / (width as usize / 2) * 2 + spatial % 4 / 2;
        let x = tile % (width as usize / 2) * 2 + spatial % 2;
        let sy = 34.0f32 / (height - 1u32) as f32;
        let sx = 34.0f32 / (width - 1u32) as f32;
        let mut yy = y as f32 * sy;
        let mut xx = x as f32 * sx;
        if y >= height as usize / 2 { yy = 34.0f32 - (height as usize - 1 - y) as f32 * sy; }
        if x >= width as usize / 2 { xx = 34.0f32 - (width as usize - 1 - x) as f32 * sx; }
        let y0 = yy as usize;
        let x0 = xx as usize;
        let y1 = usize::min(y0 + 1, 34usize);
        let x1 = usize::min(x0 + 1, 34usize);
        let dy = yy - y0 as f32;
        let dx = xx - x0 as f32;
        let p0 = table[(y0 * 35 + x0) * hidden + col] * ((1.0f32 - dy) * (1.0f32 - dx));
        let p1 = table[(y0 * 35 + x1) * hidden + col] * ((1.0f32 - dy) * dx);
        let p2 = table[(y1 * 35 + x0) * hidden + col] * (dy * (1.0f32 - dx));
        let p3 = table[(y1 * 35 + x1) * hidden + col] * (dy * dx);
        out[(offset as usize + token) * hidden + col] = ((p0 + p1) + p2) + p3;
        let dim = inv.len() * 4;
        if col < dim {
            let mut pos = x;
            if col % (dim / 2) < dim / 4 { pos = y; }
            let angle = pos as f32 * inv[col % (dim / 4)];
            cos[(offset as usize + token) * dim + col] = angle.cos();
            sin[(offset as usize + token) * dim + col] = angle.sin();
        }
    }
}

#[ruda(launch)]
pub fn video_rope(qkv: &Array<f32>, temporal: &Array<f32>, spatial: &Array<f32>, q: &mut Array<f32>,
    k: &mut Array<f32>, v: &mut Array<f32>, seq: u32, patches: u32, side: u32,
) {
    let p = ABSOLUTE_POS as usize;
    if p < q.len() {
        let row = p / 1024;
        let component = p % 1024;
        let dim = component % 64;
        let pos = row % seq as usize;
        let mut qc = qkv[row * 3072 + component];
        let mut kc = qkv[row * 3072 + 1024 + component];
        if dim < 63 && dim % 21 < 20 {
            let section = dim / 21;
            let d = dim % 21;
            let mut position = pos / patches as usize;
            let mut frequency = temporal[d / 2];
            if section > 0 {
                frequency = spatial[d / 2];
                position = (pos % patches as usize) / side as usize;
                if section == 2 { position = pos % patches as usize % side as usize; }
            }
            let angle = position as f32 * frequency;
            let mut other = component + 1;
            let mut sign = -1.0f32;
            if d % 2 != 0 { other = component - 1; sign = 1.0f32; }
            let cv = angle.cos();
            let sv = angle.sin();
            qc = qc * cv + sign * qkv[row * 3072 + other] * sv;
            kc = kc * cv + sign * qkv[row * 3072 + 1024 + other] * sv;
        }
        q[p] = qc;
        k[p] = kc;
        v[p] = qkv[row * 3072 + 2048 + component];
    }
}

#[ruda(launch)]
pub fn varlen_attention(q: &Array<f32>, k: &Array<f32>, v: &Array<f32>, cu: &Array<i64>, out: &mut Array<f32>,
    heads: u32, scale: f32, #[comptime] dim: usize,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let mut lo = 0usize;
    let mut hi = cu.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cu[mid] as usize <= row { lo = mid; } else { hi = mid; }
    }
    let start = cu[lo] as usize;
    let length = (cu[lo + 1] - cu[lo]) as usize;
    let begin = start + length * warp / 4;
    let end = start + length * (warp + 1) / 4;
    let mut query = Array::<f32>::new(dim.div_ceil(32));
    let mut acc = Array::<f32>::new(dim.div_ceil(32));
    #[unroll]
    for j in 0usize..comptime!(dim.div_ceil(32)) {
        let col = lane + j * 32;
        query[j] = 0.0f32;
        if col < dim { query[j] = q[(row * heads as usize + head) * dim + col]; }
        acc[j] = 0.0f32;
    }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    for token in begin..end {
        let mut dot = 0.0f32;
        #[unroll]
        for j in 0usize..comptime!(dim.div_ceil(32)) {
            let col = lane + j * 32;
            if col < dim { dot = fma(query[j], k[(token * heads as usize + head) * dim + col], dot); }
        }
        let score = plane_sum(dot) * scale;
        let mut next = score;
        let mut prior = 0.0f32;
        if token != begin { next = f32::max(maximum, score); prior = crate::standalone::exp_fast(maximum - next); }
        let probability = crate::standalone::exp_fast(score - next);
        denominator = fma(denominator, prior, probability);
        #[unroll]
        for j in 0usize..comptime!(dim.div_ceil(32)) {
            let col = lane + j * 32;
            if col < dim { acc[j] = fma(probability, v[(token * heads as usize + head) * dim + col], acc[j] * prior); }
        }
        maximum = next;
    }
    let state = crate::attention_state::merge_warps(&mut acc, maximum, denominator,
        warp, lane, comptime!(dim.div_ceil(32) * 32), 32usize, 4usize);
    if warp == 0 {
        #[unroll]
        for j in 0usize..comptime!(dim.div_ceil(32)) {
            let col = lane + j * 32;
            if col < dim { out[(row * heads as usize + head) * dim + col] = acc[j] / state.1; }
        }
    }
}
