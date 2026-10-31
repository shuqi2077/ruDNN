//! MoE routing for the grouped tcgen05 GEMM: counting sort of (token, k) pairs
//! by expert, row / M-tile prefixes, row gather and weighted combine.
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

use super::common::rnd;

/// counts[ids[i]] += 1 for every (token, k) pair.
#[ruda(launch)]
pub fn moe_count(ids: &Array<i64>, counts: &mut Array<Atomic<u32>>, count: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let e = ids[i] as usize;
        counts[e].fetch_add(1u32);
    }
}

/// groups = [row prefix (E + 1), 128-row M-tile prefix (E + 1)]; cursor = 0.
/// Single thread (E is small).
#[ruda(launch)]
pub fn moe_offsets(counts: &Array<u32>, groups: &mut Array<u32>, cursor: &mut Array<u32>, experts: u32) {
    if UNIT_POS == 0 {
        let e_count = experts as usize;
        let mut rows = 0u32;
        let mut tiles = 0u32;
        for e in 0..e_count {
            groups[e] = rows;
            groups[e_count + 1 + e] = tiles;
            let c = counts[e];
            rows += c;
            tiles += (c + 127u32) / 128u32;
            cursor[e] = 0u32;
        }
        groups[e_count] = rows;
        groups[2 * e_count + 1] = tiles;
    }
}

/// Places pair i = (token, k) at groups[e] + cursor[e]++; sorted[pos] = token,
/// inverse[i] = pos.
#[ruda(launch)]
pub fn moe_place(ids: &Array<i64>, groups: &Array<u32>, cursor: &mut Array<Atomic<u32>>, sorted: &mut Array<u32>,
    inverse: &mut Array<u32>, count: u32, top_k: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let e = ids[i] as usize;
        let pos = groups[e] + cursor[e].fetch_add(1u32);
        sorted[pos as usize] = (i / top_k as usize) as u32;
        inverse[i] = pos;
    }
}

/// out[r] = x[index[r]] for rows of 8-element vectors (u32 indices).
#[ruda(launch)]
pub fn moe_gather<F: Float>(x: &Array<Vector<F, Const<8>>>, index: &Array<u32>, out: &mut Array<Vector<F, Const<8>>>,
    count: u32, vectors: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let r = i / vectors as usize;
        out[i] = x[index[r] as usize * vectors as usize + i % vectors as usize];
    }
}

/// out[t] = F(sum_k p_k) with p_k = y[inverse[t*K + k]] * w[t, k] (rounded to F
/// when `round_products`), plus an optional residual row (F, added last).
#[ruda(launch)]
pub fn moe_combine<F: Float, W: Float>(y: &Array<Vector<F, Const<8>>>, inverse: &Array<u32>, weights: &Array<W>,
    residual: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<F, Const<8>>>, count: u32, vectors: u32, top_k: u32,
    #[define(F)] _f: StorageType, #[define(W)] _w: StorageType,
    #[comptime] round_products: bool, #[comptime] add_residual: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let t = i / vectors as usize;
        let v = i % vectors as usize;
        let mut acc = Vector::<f32, Const<8>>::new(0.0f32);
        for k in 0..top_k as usize {
            let position = inverse[t * top_k as usize + k];
            if position != 0xFFFFFFFFu32 {
                let w = f32::cast_from(weights[t * top_k as usize + k]);
                let row = position as usize;
                let value = Vector::<f32, Const<8>>::cast_from(y[row * vectors as usize + v]);
                #[unroll]
                for d in 0usize..8 {
                    let mut p = value[d] * w;
                    if comptime!(round_products) { p = rnd::<F>(p); }
                    acc[d] = acc[d] + p;
                }
            }
        }
        if comptime!(add_residual) { acc = acc + Vector::<f32, Const<8>>::cast_from(residual[i]); }
        out[i] = Vector::<F, Const<8>>::cast_from(acc);
    }
}

/// Stable counting sort pass 1 (keys < 256, chunks of 1024): hist[chunk][e].
/// Thread e scans its block's chunk in order.
#[ruda(launch)]
pub fn sort_hist(keys: &Array<i32>, hist: &mut Array<u32>, count: u32) {
    let chunk = RUDA_POS_X as usize;
    let e = UNIT_POS as usize;
    let mut tile = SharedMemory::<i32>::new(1024usize);
    #[unroll]
    for j in 0usize..4 {
        let i = chunk * 1024 + j * 256 + e;
        let mut v = -1i32;
        if i < count as usize { v = keys[i]; }
        tile[j * 256 + e] = v;
    }
    sync_ruda();
    let mut c = 0u32;
    for j in 0..1024usize { if tile[j] == e as i32 { c += 1u32; } }
    hist[chunk * 256 + e] = c;
}

/// Pass 2 (one block of 256): base[chunk][e] = offsets[e] + sum of hist[< chunk][e];
/// offsets (E + 1) is the exclusive prefix of the totals.
#[ruda(launch)]
pub fn sort_scan(hist: &Array<u32>, base: &mut Array<u32>, offsets: &mut Array<u32>, chunks: u32) {
    let e = UNIT_POS as usize;
    let mut totals = SharedMemory::<u32>::new(256usize);
    let mut running = 0u32;
    for c in 0..chunks as usize {
        base[c * 256 + e] = running;
        running += hist[c * 256 + e];
    }
    totals[e] = running;
    sync_ruda();
    let mut start = 0u32;
    for k in 0..e { start += totals[k]; }
    offsets[e] = start;
    if e == 255 { offsets[256] = start + running; }
    for c in 0..chunks as usize { base[c * 256 + e] = base[c * 256 + e] + start; }
}

/// Pass 3: sorted[base[chunk][e] + rank] = index for keys == e, in index order.
#[ruda(launch)]
pub fn sort_scatter(keys: &Array<i32>, base: &Array<u32>, sorted: &mut Array<i32>, count: u32) {
    let chunk = RUDA_POS_X as usize;
    let e = UNIT_POS as usize;
    let mut tile = SharedMemory::<i32>::new(1024usize);
    #[unroll]
    for j in 0usize..4 {
        let i = chunk * 1024 + j * 256 + e;
        let mut v = -1i32;
        if i < count as usize { v = keys[i]; }
        tile[j * 256 + e] = v;
    }
    sync_ruda();
    let mut position = base[chunk * 256 + e];
    for j in 0..1024usize {
        if tile[j] == e as i32 {
            sorted[position as usize] = (chunk * 1024 + j) as i32;
            position += 1u32;
        }
    }
}

/// Capacity-limited routing from the stable sort (sorted pair indices + offsets
/// (E + 1)): keeps the first `capacity` pairs of every expert. groups = [kept row
/// prefix (E + 1), M-tile prefix (E + 1)] (single thread), see `cap_place`.
#[ruda(launch)]
pub fn cap_offsets(offsets: &Array<u32>, groups: &mut Array<u32>, experts: u32, capacity: u32) {
    if UNIT_POS == 0 {
        let e_count = experts as usize;
        let mut rows = 0u32;
        let mut tiles = 0u32;
        for e in 0..e_count {
            groups[e] = rows;
            groups[e_count + 1 + e] = tiles;
            let mut c = offsets[e + 1] - offsets[e];
            if c > capacity { c = capacity; }
            rows += c;
            tiles += (c + 127u32) / 128u32;
        }
        groups[e_count] = rows;
        groups[2 * e_count + 1] = tiles;
    }
}

/// For sorted position i (pair sorted[i], expert e): kept rows go to
/// groups[e] + (i - offsets[e]); tokens[row] = pair / top_k, inverse[pair] = row
/// (0xFFFFFFFF when dropped).
#[ruda(launch)]
pub fn cap_place(sorted: &Array<i32>, offsets: &Array<u32>, groups: &Array<u32>, tokens: &mut Array<u32>,
    inverse: &mut Array<u32>, count: u32, experts: u32, capacity: u32, top_k: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let mut lo = 0usize;
        let mut hi = experts as usize;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if offsets[mid] as usize <= i { lo = mid; } else { hi = mid; }
        }
        let within = i as u32 - offsets[lo];
        let pair = sorted[i] as usize;
        if within < capacity {
            let row = groups[lo] + within;
            tokens[row as usize] = (pair / top_k as usize) as u32;
            inverse[pair] = row;
        } else {
            inverse[pair] = 0xFFFFFFFFu32;
        }
    }
}

/// out[i] = i32(ids[i]).
#[ruda(launch)]
pub fn narrow_ids(ids: &Array<i64>, out: &mut Array<i32>, count: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize { out[i] = ids[i] as i32; }
}
