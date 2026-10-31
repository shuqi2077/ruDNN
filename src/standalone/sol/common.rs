//! Generic RUDA building blocks shared by the SOL L1/L2/Quant solutions:
//! 3xTF32 operand split, row norms, row softmax, RoPE, im2col, gathers,
//! reductions, casts and broadcast binary ops. All accumulate in f32 and
//! round at the PyTorch rounding points of the reference ops.
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

use crate::standalone::block_sum;

#[ruda]
pub fn rnd<F: Float>(value: f32) -> f32 {
    f32::cast_from(F::cast_from(value))
}

/// 3xTF32 operand: each `chunk` input elements become 3 output chunks of
/// `out_chunk` (>= chunk, zero padded): [hi, lo, hi] (A side) or [hi, hi, lo].
/// `count` is the number of output elements.
#[ruda(launch)]
pub fn tf32_split(x: &Array<f32>, out: &mut Array<f32>, count: u32, chunk: u32, out_chunk: u32, in_stride: u32,
    #[comptime] a_side: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let oc = out_chunk as usize;
        let r = i / (3 * oc);
        let part = (i / oc) % 3;
        let c = i % oc;
        let mut v = 0.0f32;
        if c < chunk as usize { v = x[r * in_stride as usize + c]; }
        let hi = f32::cast_from(tf32::cast_from(v));
        let mut take_lo = part == 2;
        if comptime!(a_side) { take_lo = part == 1; }
        let mut value = hi;
        if take_lo { value = v - hi; }
        out[i] = value;
    }
}

#[ruda]
fn norm_load<F: Float>(x: &Array<Vector<F, Const<8>>>, residual: &Array<Vector<F, Const<8>>>,
    residual_out: &mut Array<Vector<F, Const<8>>>, index: usize, #[comptime] add: bool, #[comptime] first: bool,
) -> Vector<f32, Const<8>> {
    let mut value = Vector::<f32, Const<8>>::cast_from(x[index]);
    if comptime!(add) {
        if comptime!(first) {
            let sum = Vector::<F, Const<8>>::cast_from(value + Vector::<f32, Const<8>>::cast_from(residual[index]));
            residual_out[index] = sum;
            value = Vector::<f32, Const<8>>::cast_from(sum);
        } else {
            value = Vector::<f32, Const<8>>::cast_from(residual_out[index]);
        }
    }
    value
}

#[ruda]
fn norm_cached<F: Float, W: Float, O: Float>(x: &Array<Vector<F, Const<8>>>, residual: &Array<Vector<F, Const<8>>>,
    weight: &Array<Vector<W, Const<8>>>, bias: &Array<Vector<W, Const<8>>>,
    out: &mut Array<Vector<O, Const<8>>>, residual_out: &mut Array<Vector<F, Const<8>>>, width: u32, eps: f32,
    #[comptime] layer: bool, #[comptime] add: bool, #[comptime] use_weight: bool, #[comptime] use_bias: bool,
    #[comptime] one_plus: bool, #[comptime] cast: bool,
) {
    let thread = UNIT_POS as usize;
    let vectors = width as usize / 8;
    let base = RUDA_POS_X as usize * vectors;
    let mut cache = Array::<Vector<f32, Const<8>>>::new(4usize);
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..4 {
        let col = thread + j * 256;
        let mut v = Vector::<f32, Const<8>>::new(0.0f32);
        if col < vectors {
            v = norm_load(x, residual, residual_out, base + col, add, true);
            #[unroll]
            for d in 0usize..8 {
                if comptime!(layer) { local += v[d]; } else { local += v[d] * v[d]; }
            }
        }
        cache[j] = v;
    }
    let total = block_sum(local, &mut scratch, 256usize);
    let mut mean = 0.0f32;
    let mut squares = total;
    if comptime!(layer) {
        mean = total / width as f32;
        let mut local_squares = 0.0f32;
        #[unroll]
        for j in 0usize..4 {
            if thread + j * 256 < vectors {
                let v = cache[j];
                #[unroll]
                for d in 0usize..8 { let c = v[d] - mean; local_squares += c * c; }
            }
        }
        squares = block_sum(local_squares, &mut scratch, 256usize);
    }
    let inv = (squares / width as f32 + eps).inverse_sqrt();
    #[unroll]
    for j in 0usize..4 {
        let col = thread + j * 256;
        if col < vectors {
            let v = cache[j];
            let mut w = Vector::<f32, Const<8>>::new(1.0f32);
            if comptime!(use_weight) { w = Vector::<f32, Const<8>>::cast_from(weight[col]); }
            let mut b = Vector::<f32, Const<8>>::new(0.0f32);
            if comptime!(use_bias) { b = Vector::<f32, Const<8>>::cast_from(bias[col]); }
            let mut result = Vector::<O, Const<8>>::empty();
            #[unroll]
            for d in 0usize..8 {
                let mut n = (v[d] - mean) * inv;
                if comptime!(cast) { n = rnd::<F>(n); }
                if comptime!(use_weight) {
                    if comptime!(one_plus) { n = n * (1.0f32 + w[d]); } else { n = n * w[d]; }
                }
                if comptime!(use_bias) { n += b[d]; }
                result[d] = O::cast_from(n);
            }
            out[base + col] = result;
        }
    }
}

/// Row RMSNorm (`layer` = false) or LayerNorm, 256 threads per row, runtime width
/// (multiple of 8). `add`: h = F(x + residual) is normalized and written to
/// residual_out. `cast`: normalized value rounded to F before the weight
/// (HF `weight * h.to(dtype)`); `one_plus`: weight is (1 + w) in f32.
#[ruda(launch)]
pub fn norm_rows<F: Float, W: Float, O: Float>(x: &Array<Vector<F, Const<8>>>, residual: &Array<Vector<F, Const<8>>>,
    weight: &Array<Vector<W, Const<8>>>, bias: &Array<Vector<W, Const<8>>>,
    out: &mut Array<Vector<O, Const<8>>>, residual_out: &mut Array<Vector<F, Const<8>>>,
    width: u32, eps: f32,
    #[define(F)] _f: StorageType, #[define(W)] _w: StorageType, #[define(O)] _o: StorageType,
    #[comptime] layer: bool, #[comptime] add: bool, #[comptime] use_weight: bool, #[comptime] use_bias: bool,
    #[comptime] one_plus: bool, #[comptime] cast: bool,
) {
    if width <= 8192u32 {
        norm_cached(x, residual, weight, bias, out, residual_out, width, eps,
            layer, add, use_weight, use_bias, one_plus, cast);
        terminate!();
    }
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let vectors = width as usize / 8;
    let base = row * vectors;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < vectors {
        let v = norm_load(x, residual, residual_out, base + j, add, true);
        #[unroll]
        for d in 0usize..8 {
            if comptime!(layer) { local += v[d]; } else { local += v[d] * v[d]; }
        }
        j += 256;
    }
    let total = block_sum(local, &mut scratch, 256usize);
    let mut mean = 0.0f32;
    let mut inv = 0.0f32;
    if comptime!(layer) {
        mean = total / width as f32;
        let mut squares = 0.0f32;
        let mut j2 = thread;
        while j2 < vectors {
            let v = norm_load(x, residual, residual_out, base + j2, add, false);
            #[unroll]
            for d in 0usize..8 { let c = v[d] - mean; squares += c * c; }
            j2 += 256;
        }
        inv = (block_sum(squares, &mut scratch, 256usize) / width as f32 + eps).inverse_sqrt();
    } else {
        inv = (total / width as f32 + eps).inverse_sqrt();
    }
    let mut j3 = thread;
    while j3 < vectors {
        let v = norm_load(x, residual, residual_out, base + j3, add, false);
        let mut w = Vector::<f32, Const<8>>::new(1.0f32);
        if comptime!(use_weight) { w = Vector::<f32, Const<8>>::cast_from(weight[j3]); }
        let mut b = Vector::<f32, Const<8>>::new(0.0f32);
        if comptime!(use_bias) { b = Vector::<f32, Const<8>>::cast_from(bias[j3]); }
        let mut result = Vector::<O, Const<8>>::empty();
        #[unroll]
        for d in 0usize..8 {
            let mut n = (v[d] - mean) * inv;
            if comptime!(cast) { n = rnd::<F>(n); }
            if comptime!(use_weight) {
                if comptime!(one_plus) { n = n * (1.0f32 + w[d]); } else { n = n * w[d]; }
            }
            if comptime!(use_bias) { n = n + b[d]; }
            result[d] = O::cast_from(n);
        }
        out[base + j3] = result;
        j3 += 256;
    }
}

/// Row softmax over `columns` with optional additive (f32) / boolean (u8) mask
/// and causal masking; 256 threads per row, f32 math, O output.
/// mask row = (row / outer) * inner + row % inner; query = row % queries.
#[ruda(launch)]
pub fn softmax_rows<F: Float, O: Float>(x: &Array<F>, mask: &Array<f32>, flags: &Array<u8>, out: &mut Array<O>,
    columns: u32, ld_in: u32, ld_out: u32, scale: f32, outer: u32, inner: u32, queries: u32, offset: u32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
    #[comptime] mask_mode: u32, #[comptime] causal: bool,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = columns as usize;
    let base = row * ld_in as usize;
    let out_base = row * ld_out as usize;
    let mask_base = ((row / outer as usize) * inner as usize + row % inner as usize) * n;
    // (padding columns of the output row, if any, are left untouched)
    let query = row % queries as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = f32::new(f32::NEG_INFINITY);
    let mut j = thread;
    while j < n {
        let s = softmax_score(x, mask, flags, base, mask_base, j, query, offset, scale, mask_mode, causal);
        local = f32::max(local, s);
        j += 256;
    }
    // block max
    let mut m = plane_max(local);
    if thread % 32 == 0 { scratch[thread / 32] = m; }
    sync_ruda();
    let mut part = f32::new(f32::NEG_INFINITY);
    if thread % 32 < 8 { part = scratch[thread % 32]; }
    m = plane_max(part);
    sync_ruda();
    let mut used = m;
    if m == f32::new(f32::NEG_INFINITY) { used = 0.0f32; }
    if n <= 4096usize {
        let mut cache = Array::<f32>::new(16usize);
        let mut local_sum = 0.0f32;
        #[unroll]
        for slot in 0usize..16 {
            let col = thread + slot * 256;
            let mut probability = 0.0f32;
            if col < n {
                let s = softmax_score(x, mask, flags, base, mask_base, col, query, offset, scale, mask_mode, causal);
                probability = crate::standalone::exp_fast(s - used);
                local_sum += probability;
            }
            cache[slot] = probability;
        }
        let inverse = 1.0f32 / block_sum(local_sum, &mut scratch, 256usize);
        #[unroll]
        for slot in 0usize..16 {
            let col = thread + slot * 256;
            if col < n { out[out_base + col] = O::cast_from(cache[slot] * inverse); }
        }
        terminate!();
    }
    let mut sum = 0.0f32;
    let mut j2 = thread;
    while j2 < n {
        let s = softmax_score(x, mask, flags, base, mask_base, j2, query, offset, scale, mask_mode, causal);
        sum += crate::standalone::exp_fast(s - used);
        j2 += 256;
    }
    let total = block_sum(sum, &mut scratch, 256usize);
    let inverse = 1.0f32 / total;
    let mut j3 = thread;
    while j3 < n {
        let s = softmax_score(x, mask, flags, base, mask_base, j3, query, offset, scale, mask_mode, causal);
        out[out_base + j3] = O::cast_from(crate::standalone::exp_fast(s - used) * inverse);
        j3 += 256;
    }
}

#[ruda]
fn softmax_score<F: Float>(x: &Array<F>, mask: &Array<f32>, flags: &Array<u8>, base: usize, mask_base: usize,
    column: usize, query: usize, offset: u32, scale: f32, #[comptime] mask_mode: u32, #[comptime] causal: bool,
) -> f32 {
    let mut s = f32::cast_from(x[base + column]) * scale;
    if comptime!(mask_mode == 1u32) { s += mask[mask_base + column]; }
    let mut valid = true;
    if comptime!(mask_mode == 2u32) { valid = flags[mask_base + column] != 0u8; }
    if comptime!(causal) { if column > query + offset as usize { valid = false; } }
    if !valid { s = f32::new(f32::NEG_INFINITY); }
    s
}

/// RoPE on rows of `dim` (rotate_half or interleaved pairs) over the first
/// `rotary` dims; position = (row / div) % modulo indexes cos/sin (P, rotary).
/// `exact`: every op rounded through F (bf16 reference arithmetic).
#[ruda(launch)]
pub fn rope_rows<F: Float, C: Float>(x: &Array<F>, cos: &Array<C>, sin: &Array<C>, out: &mut Array<F>,
    count: u32, dim: u32, rotary: u32, div: u32, modulo: u32,
    #[define(F)] _f: StorageType, #[define(C)] _c: StorageType,
    #[comptime] interleaved: bool, #[comptime] exact: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let d = i % dim as usize;
        let row = i / dim as usize;
        let r = rotary as usize;
        if d >= r {
            out[i] = x[i];
        } else {
            let position = (row / div as usize) % modulo as usize;
            let mut partner = 0.0f32;
            let mut c = 0.0f32;
            let mut s = 0.0f32;
            if comptime!(interleaved) {
                if d % 2 == 0 { partner = -f32::cast_from(x[i + 1]); } else { partner = f32::cast_from(x[i - 1]); }
                c = f32::cast_from(cos[position * r + d]);
                s = f32::cast_from(sin[position * r + d]);
            } else {
                let half = r / 2;
                if d < half { partner = -f32::cast_from(x[i + half]); } else { partner = f32::cast_from(x[i - half]); }
                c = f32::cast_from(cos[position * r + d]);
                s = f32::cast_from(sin[position * r + d]);
            }
            let v = f32::cast_from(x[i]);
            let mut a = v * c;
            let mut b = partner * s;
            if comptime!(exact) { a = rnd::<F>(a); b = rnd::<F>(b); }
            out[i] = F::cast_from(a + b);
        }
    }
}

/// im2col for NCHW conv2d: cols (N, OH*OW, kp) with k = (c*KH + kh)*KW + kw,
/// zero padded to kp columns.
#[ruda(launch)]
pub fn im2col<F: Float>(input: &Array<F>, cols: &mut Array<F>, count: u32,
    channels: u32, height: u32, width: u32, kh: u32, kw: u32, oh: u32, ow: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32, dil_h: u32, dil_w: u32, kp: u32,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let k = i % kp as usize;
        let p = (i / kp as usize) % (oh * ow) as usize;
        let n = i / (kp as usize * (oh * ow) as usize);
        let mut value = F::cast_from(0.0f32);
        if k < (channels * kh * kw) as usize {
            let c = k / (kh * kw) as usize;
            let r = (k / kw as usize) % kh as usize;
            let s = k % kw as usize;
            let y = (p / ow as usize * stride_h as usize + r * dil_h as usize) as i32 - pad_h as i32;
            let xx = (p % ow as usize * stride_w as usize + s * dil_w as usize) as i32 - pad_w as i32;
            if y >= 0 && y < height as i32 && xx >= 0 && xx < width as i32 {
                value = input[((n * channels as usize + c) * height as usize + y as usize) * width as usize + xx as usize];
            }
        }
        cols[i] = value;
    }
}

/// out[r] = table[index[r]] for rows of 8-element vectors.
#[ruda(launch)]
pub fn gather_rows<F: Float>(table: &Array<Vector<F, Const<8>>>, index: &Array<i64>, out: &mut Array<Vector<F, Const<8>>>,
    count: u32, vectors: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let row = i / vectors as usize;
        out[i] = table[index[row] as usize * vectors as usize + i % vectors as usize];
    }
}

/// Column partial sums: partial[chunk][col] = sum over rows of the chunk.
#[ruda(launch)]
pub fn col_partial<F: Float>(x: &Array<F>, partial: &mut Array<f32>, rows: u32, columns: u32, chunk: u32,
    #[define(F)] _f: StorageType,
) {
    let col = RUDA_POS_X as usize * 256 + UNIT_POS as usize;
    let part = RUDA_POS_Y as usize;
    if col < columns as usize {
        let begin = part * chunk as usize;
        let mut end = begin + chunk as usize;
        if end > rows as usize { end = rows as usize; }
        let mut total = 0.0f32;
        let mut r = begin;
        while r < end { total += f32::cast_from(x[r * columns as usize + col]); r += 1; }
        partial[part * columns as usize + col] = total;
    }
}

/// Column sum of partials (chunks, columns) -> out (columns) as O.
#[ruda(launch)]
pub fn col_finish<O: Float>(partial: &Array<f32>, out: &mut Array<O>, columns: u32, chunks: u32, #[define(O)] _o: StorageType) {
    let col = ABSOLUTE_POS as usize;
    if col < columns as usize {
        let mut total = 0.0f32;
        for p in 0..chunks as usize { total += partial[p * columns as usize + col]; }
        out[col] = O::cast_from(total);
    }
}

/// Sum of each row (256 threads per row) -> out[row] as O.
#[ruda(launch)]
pub fn row_sum<F: Float, O: Float>(x: &Array<F>, out: &mut Array<O>, columns: u32, scale: f32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < columns as usize { local += f32::cast_from(x[row * columns as usize + j]); j += 256; }
    let total = block_sum(local, &mut scratch, 256usize);
    if thread == 0 { out[row] = O::cast_from(total * scale); }
}

/// Elementwise cast F -> O over 8-element vectors.
#[ruda(launch)]
pub fn cast_vec<F: Float, O: Float>(x: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<O, Const<8>>>, count: u32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize { out[i] = Vector::<O, Const<8>>::cast_from(x[i]); }
}

/// 4D strided copy into a contiguous (d0, d1, d2, d3) output.
#[ruda(launch)]
pub fn permute_copy<F: Float>(x: &Array<F>, out: &mut Array<F>, d1: u32, d2: u32, d3: u32, count: u32,
    s0: u32, s1: u32, s2: u32, s3: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let a3 = i % d3 as usize;
        let a2 = (i / d3 as usize) % d2 as usize;
        let a1 = (i / (d3 * d2) as usize) % d1 as usize;
        let a0 = i / (d3 * d2 * d1) as usize;
        out[i] = x[a0 * s0 as usize + a1 * s1 as usize + a2 * s2 as usize + a3 * s3 as usize];
    }
}

/// Broadcast binary op over a 4D output (d0, d1, d2, d3); strides per operand
/// (0 = broadcast). op: 0 add, 1 sub, 2 mul, 3 div, 4 max, 5 min. Single
/// rounding to O (PyTorch semantics for one op).
#[ruda(launch)]
pub fn binary4<F: Float, G: Float, O: Float>(a: &Array<F>, b: &Array<G>, out: &mut Array<O>,
    d1: u32, d2: u32, d3: u32, count: u32,
    a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32, b2: u32, b3: u32, alpha: f32,
    #[define(F)] _f: StorageType, #[define(G)] _g: StorageType, #[define(O)] _o: StorageType,
    #[comptime] op: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let i3 = i % d3 as usize;
        let i2 = (i / d3 as usize) % d2 as usize;
        let i1 = (i / (d3 * d2) as usize) % d1 as usize;
        let i0 = i / (d3 * d2 * d1) as usize;
        let x = f32::cast_from(a[i0 * a0 as usize + i1 * a1 as usize + i2 * a2 as usize + i3 * a3 as usize]);
        let y = f32::cast_from(b[i0 * b0 as usize + i1 * b1 as usize + i2 * b2 as usize + i3 * b3 as usize]);
        let mut r = x + alpha * y;
        if comptime!(op == 1u32) { r = x - alpha * y; }
        else if comptime!(op == 2u32) { r = x * y; }
        else if comptime!(op == 3u32) { r = x / y; }
        else if comptime!(op == 4u32) { r = f32::max(x, y); }
        else if comptime!(op == 5u32) { r = f32::min(x, y); }
        out[i] = O::cast_from(r);
    }
}

/// Per-head RMSNorm + rotate_half RoPE for rows of `dim` (64 / 128 / 256): one warp
/// per row, 8 rows per 256-thread block; lane l owns the pairs (d, d + dim/2) for
/// d = l + 32 j. cos/sin row = (row / outer) * inner + row % inner; `computed`:
/// cos/sin = F(cos/sin(position[crow] * inv_freq[d % (dim/2)])), else tables of C.
/// norm: 0 none, 1 HF (w * F(n), product in F), 2 F(f32(w) * n).
/// `exact`: RoPE products and sum rounded through F (bf16 reference arithmetic).
#[ruda(launch)]
pub fn head_norm_rope<F: Float, W: Float, C: Float>(x: &Array<F>, weight: &Array<W>, bias: &Array<W>, cos: &Array<C>, sin: &Array<C>,
    positions: &Array<i64>, inv_freq: &Array<f32>, out: &mut Array<F>, rows: u32, outer: u32, inner: u32, stride: u32, eps: f32,
    #[define(F)] _f: StorageType, #[define(W)] _w: StorageType, #[define(C)] _c: StorageType,
    #[comptime] dim: usize, #[comptime] norm: u32, #[comptime] rope: bool, #[comptime] computed: bool, #[comptime] exact: bool,
    #[comptime] half_table: bool,
) {
    let row = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if row < rows as usize {
        let half = comptime!(dim / 2);
        let pairs = comptime!((dim / 2 + 31) / 32);
        let table = comptime!(if half_table { dim / 2 } else { dim });
        let base = row * stride as usize;
        let mut lo = Array::<f32>::new(pairs);
        let mut hi = Array::<f32>::new(pairs);
        let mut squares = 0.0f32;
        #[unroll]
        for j in 0usize..pairs {
            let d = lane + 32 * j;
            lo[j] = 0.0f32;
            hi[j] = 0.0f32;
            if d < half {
                lo[j] = f32::cast_from(x[base + d]);
                hi[j] = f32::cast_from(x[base + d + half]);
            }
            squares += lo[j] * lo[j] + hi[j] * hi[j];
        }
        if comptime!(norm == 4u32) {
            // LayerNorm with weight and bias (F(((x - mean) / sqrt(var + eps)) * w + b)).
            let mut total = 0.0f32;
            #[unroll]
            for j in 0usize..pairs { total += lo[j] + hi[j]; }
            let mean = plane_sum(total) / dim as f32;
            let mut sq = 0.0f32;
            #[unroll]
            for j in 0usize..pairs {
                let d = lane + 32 * j;
                if d < half { let a = lo[j] - mean; let c = hi[j] - mean; sq += a * a + c * c; }
            }
            let inv = 1.0f32 / (plane_sum(sq) / dim as f32 + eps).sqrt();
            #[unroll]
            for j in 0usize..pairs {
                let d = (lane + 32 * j) % half;
                lo[j] = rnd::<F>((lo[j] - mean) * inv * f32::cast_from(weight[d]) + f32::cast_from(bias[d]));
                hi[j] = rnd::<F>((hi[j] - mean) * inv * f32::cast_from(weight[d + half]) + f32::cast_from(bias[d + half]));
            }
        } else if comptime!(norm != 0u32) {
            let inv = (plane_sum(squares) / dim as f32 + eps).inverse_sqrt();
            #[unroll]
            for j in 0usize..pairs {
                let d = (lane + 32 * j) % half;
                let wl = f32::cast_from(weight[d]);
                let wh = f32::cast_from(weight[d + half]);
                if comptime!(norm == 1u32) {
                    lo[j] = rnd::<F>(wl * rnd::<F>(lo[j] * inv));
                    hi[j] = rnd::<F>(wh * rnd::<F>(hi[j] * inv));
                } else if comptime!(norm == 3u32) {
                    lo[j] = rnd::<F>((lo[j] * inv) * (1.0f32 + wl));
                    hi[j] = rnd::<F>((hi[j] * inv) * (1.0f32 + wh));
                } else if comptime!(norm == 5u32) {
                    lo[j] = rnd::<F>(lo[j] * inv);
                    hi[j] = rnd::<F>(hi[j] * inv);
                } else {
                    lo[j] = rnd::<F>(wl * (lo[j] * inv));
                    hi[j] = rnd::<F>(wh * (hi[j] * inv));
                }
            }
        }
        let crow = (row / outer as usize) * inner as usize + row % inner as usize;
        #[unroll]
        for j in 0usize..pairs {
            let d_raw = lane + 32 * j;
            let d = d_raw % half;
            let mut a = lo[j];
            let mut b = hi[j];
            if comptime!(rope) {
                let mut cl = 0.0f32;
                let mut sl = 0.0f32;
                let mut ch = 0.0f32;
                let mut sh = 0.0f32;
                if comptime!(computed) {
                    let angle = f32::cast_from(positions[crow]) * inv_freq[d];
                    cl = angle.cos();
                    sl = angle.sin();
                    if comptime!(exact) { cl = rnd::<F>(cl); sl = rnd::<F>(sl); }
                    ch = cl;
                    sh = sl;
                } else if comptime!(half_table) {
                    cl = f32::cast_from(cos[crow * table + d]);
                    sl = f32::cast_from(sin[crow * table + d]);
                    ch = cl;
                    sh = sl;
                } else {
                    cl = f32::cast_from(cos[crow * table + d]);
                    sl = f32::cast_from(sin[crow * table + d]);
                    ch = f32::cast_from(cos[crow * table + d + half]);
                    sh = f32::cast_from(sin[crow * table + d + half]);
                }
                let mut p0 = lo[j] * cl;
                let mut p1 = -hi[j] * sl;
                let mut p2 = hi[j] * ch;
                let mut p3 = lo[j] * sh;
                if comptime!(exact) { p0 = rnd::<F>(p0); p1 = rnd::<F>(p1); p2 = rnd::<F>(p2); p3 = rnd::<F>(p3); }
                a = p0 + p1;
                b = p2 + p3;
            }
            if d_raw < half {
                out[base + d] = F::cast_from(a);
                out[base + d + half] = F::cast_from(b);
            }
        }
    }
}

/// dst[(b*H + h), index[s], :] = src[(b*H + h), s, :] (rows of 8-vectors).
#[ruda(launch)]
pub fn scatter_seq<F: Float>(src: &Array<Vector<F, Const<8>>>, index: &Array<i64>, dst: &mut Array<Vector<F, Const<8>>>,
    count: u32, seq: u32, length: u32, vectors: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let v = i % vectors as usize;
        let s = (i / vectors as usize) % seq as usize;
        let bh = i / (vectors as usize * seq as usize);
        let position = index[s] as usize;
        dst[(bh * length as usize + position) * vectors as usize + v] = src[i];
    }
}

#[ruda]
pub fn unary_op(x: f32, #[comptime] op: u32) -> f32 {
    let mut r = x;
    if comptime!(op == 0u32) { r = x / (1.0f32 + (-x).exp()); }
    else if comptime!(op == 1u32) {
        let inner = 0.7978845608028654f32 * (x + 0.044715f32 * (x * x * x));
        r = 0.5f32 * x * (1.0f32 + inner.tanh());
    }
    else if comptime!(op == 2u32) { r = x * 0.5f32 * (1.0f32 + (x * 0.7071067811865476f32).erf()); }
    else if comptime!(op == 3u32) { r = 1.0f32 / (1.0f32 + (-x).exp()); }
    else if comptime!(op == 4u32) { r = x.tanh(); }
    else if comptime!(op == 5u32) { if x < 0.0f32 { r = 0.0f32; } }
    else if comptime!(op == 6u32) { r = x.exp(); }
    else if comptime!(op == 7u32) { r = x * x; }
    else if comptime!(op == 8u32) { r = x.sin(); }
    else if comptime!(op == 9u32) { r = x.cos(); }
    else if comptime!(op == 10u32) { r = x.sqrt(); }
    else if comptime!(op == 11u32) { r = 1.0f32 / x.sqrt(); }
    else if comptime!(op == 12u32) { r = x.abs(); }
    else if comptime!(op == 13u32) { r = -x; }
    else if comptime!(op == 14u32) {
        let t = (0.7978845608028654f32 * (x + 0.044715f32 * x * x * x)).tanh();
        r = 0.5f32 * (1.0f32 + t) + 0.5f32 * x * (1.0f32 - t * t) * 0.7978845608028654f32 * (1.0f32 + 0.134145f32 * x * x);
    }
    else if comptime!(op == 15u32) { r = 0.0f32; if x > 0.0f32 { r = 1.0f32; } }
    else if comptime!(op == 16u32) {
        let s = 1.0f32 / (1.0f32 + (-x).exp());
        r = s * (1.0f32 + x * (1.0f32 - s));
    }
    else if comptime!(op == 17u32) { r = 1.0f32 - x * x; }
    else if comptime!(op == 18u32) { r = 1.0f32 / x; }
    else if comptime!(op == 19u32) { r = x.ln(); }
    else if comptime!(op == 21u32) { r = x; if x <= 20.0f32 { r = (1.0f32 + x.exp()).ln(); } }
    r
}

/// Elementwise unary op (0 silu, 1 gelu_tanh, 2 gelu_erf, 3 sigmoid, 4 tanh,
/// 5 relu, 6 exp, 7 square), F -> O with a single rounding.
#[ruda(launch)]
pub fn unary<F: Float, O: Float>(x: &Array<F>, out: &mut Array<O>, count: u32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType, #[comptime] op: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize { out[i] = O::cast_from(unary_op(f32::cast_from(x[i]), op)); }
}

#[ruda(launch)]
pub fn affine_scalar<F: Float, O: Float>(x: &Array<F>, out: &mut Array<O>, scale: f32, bias: f32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() { out[i] = O::cast_from(f32::cast_from(x[i]) * scale + bias); }
}

#[ruda(launch)]
pub fn reduce_last<F: Float, O: Float>(x: &Array<F>, out: &mut Array<O>, width: u32, scale: f32,
    #[define(F)] _f: StorageType, #[define(O)] _o: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut total = 0.0f32;
    let mut j = UNIT_POS as usize;
    while j < width as usize { total += f32::cast_from(x[row * width as usize + j]); j += 256; }
    let sum = block_sum(total, &mut scratch, 256usize);
    if UNIT_POS == 0u32 { out[row] = O::cast_from(sum * scale); }
}

/// Adaptive LayerNorm rows (no affine): out = n * (1 + scale[b]) + shift[b] with
/// b = row / per_batch; shift / scale rows live in `modulation` (B, mod_width)
/// at element offsets shift_off / scale_off. 256 threads per row.
#[ruda(launch)]
pub fn adaln_rows<F: Float, M: Float, O: Float>(x: &Array<F>, modulation: &Array<M>, out: &mut Array<O>,
    width: u32, per_batch: u32, mod_width: u32, shift_off: u32, scale_off: u32, eps: f32,
    #[define(F)] _f: StorageType, #[define(M)] _m: StorageType, #[define(O)] _o: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = width as usize;
    let base = row * n;
    let mbase = (row / per_batch as usize) * mod_width as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut j = thread;
    while j < n { local += f32::cast_from(x[base + j]); j += 256; }
    let mean = block_sum(local, &mut scratch, 256usize) / n as f32;
    let mut squares = 0.0f32;
    let mut j2 = thread;
    while j2 < n { let c = f32::cast_from(x[base + j2]) - mean; squares += c * c; j2 += 256; }
    let inv = 1.0f32 / (block_sum(squares, &mut scratch, 256usize) / n as f32 + eps).sqrt();
    let mut j3 = thread;
    while j3 < n {
        let v = (f32::cast_from(x[base + j3]) - mean) * inv;
        let sc = f32::cast_from(modulation[mbase + scale_off as usize + j3]);
        let sh = f32::cast_from(modulation[mbase + shift_off as usize + j3]);
        out[base + j3] = O::cast_from(v * (1.0f32 + sc) + sh);
        j3 += 256;
    }
}
