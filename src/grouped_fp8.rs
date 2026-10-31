use ruda_core::e4m3;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

pub mod blackwell;

#[ruda]
pub(crate) fn tile_table(counts: &Array<u32>, table: &mut SharedMemory<u32>,
    #[comptime] tile_rows: usize,
) {
    let lane = UNIT_POS as usize;
    if lane < 32 {
        let count = counts[lane];
        let tiles = (count + tile_rows as u32 - 1) / tile_rows as u32;
        table[lane] = plane_exclusive_sum(tiles);
        let total = plane_sum(tiles);
        if lane == 0 { table[32usize] = total; }
        table[33usize + lane] = count;
    }
    sync_ruda();
}

#[ruda]
pub(crate) fn tile_expert(table: &SharedMemory<u32>, tile: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = 31usize;
    #[unroll]
    for _ in 0usize..5 {
        let mid = (lo + hi + 1) / 2;
        if table[mid] as usize <= tile { lo = mid; }
        else { hi = mid - 1; }
    }
    lo
}

#[ruda(launch)]
pub fn swiglu(
    input: &Array<u8>, input_scale: &Array<f32>, weights: &Array<u8>, weight_scale: &Array<f32>,
    rows: &Array<u32>, counts: &Array<u32>, offsets: &Array<u32>, output: &mut Array<f32>,
    tokens: u32, #[comptime] inner: usize, #[comptime] width: usize,
) {
    let thread = UNIT_POS as usize;
    let warp = thread / 32;
    let lane = UNIT_POS_PLANE;
    let def = cmma::MmaDefinition::<e4m3, e4m3, f32>::new(16usize, 8usize, 32usize);
    let mut table = SharedMemory::<u32>::new(65usize);
    let mut a = SharedMemory::<u8>::new_aligned(32usize * 128, 16usize);
    let mut b = SharedMemory::<u8>::new_aligned(2usize * 32 * 128, 16usize);
    tile_table(counts, &mut table, 32usize);
    let m_tiles = table[32usize] as usize;
    let n_tiles = width / 32;
    let mut tile = RUDA_POS_X as usize;
    while tile < m_tiles * n_tiles {
        let m_tile = tile % m_tiles;
        let col_base = tile / m_tiles * 32;
        let expert = tile_expert(&table, m_tile);
        let row_base = (m_tile - table[expert] as usize) * 32;
        let count = table[33usize + expert] as usize;
        let offset = offsets[expert] as usize;
        let mut sum = Array::<Vector<f32, Const<2>>>::new(8usize);
        #[unroll] for i in 0usize..8 { sum[i] = Vector::new(0.0f32); }
        for group in 0usize..(inner / 128) {
            #[unroll]
            for load in 0usize..8 {
                let p = (thread + load * 128) * 4;
                let row = row_base + p / 128;
                let mut data = Vector::<u8, Const<4>>::new(0u8);
                if row < count {
                    let token = rows[expert * tokens as usize + row] as usize;
                    let packed = input.to_slice().into_vectorized().with_vector_size::<Const<4>>();
                    data = packed[(token * inner + group * 128 + p % 128) / 4];
                }
                #[unroll] for j in 0usize..4 { a[p + j] = data[j]; }
            }
            #[unroll]
            for load in 0usize..16 {
                let p = (thread + load * 128) * 4;
                let half = p / (32 * 128);
                let col = col_base + p / 128 % 32 + half * width;
                let packed = weights.to_slice().into_vectorized().with_vector_size::<Const<4>>();
                let data = packed[((expert * width * 2 + col) * inner + group * 128 + p % 128) / 4];
                #[unroll] for j in 0usize..4 { b[p + j] = data[j]; }
            }
            sync_ruda();
            let mut acc = Array::<Vector<f32, Const<2>>>::new(8usize);
            #[unroll] for i in 0usize..8 { acc[i] = Vector::new(0.0f32); }
            #[unroll]
            for step in 0usize..4 {
                let mut left = Array::<Vector<e4m3, Const<4>>>::new(4usize);
                #[unroll]
                for i in 0usize..4 {
                    let mut value = Vector::<e4m3, Const<4>>::empty();
                    #[unroll]
                    for j in 0usize..4 {
                        let (r, c) = def.position_of_nth(lane, (i * 4 + j) as u32, cmma::MatrixIdent::A);
                        value[j] = e4m3::reinterpret(a[(warp / 2 * 16 + r as usize) * 128 + step * 32 + c as usize]);
                    }
                    left[i] = value;
                }
                #[unroll]
                for half in 0usize..2 {
                    #[unroll]
                    for column in 0usize..2 {
                        let mut right = Array::<Vector<e4m3, Const<4>>>::new(2usize);
                        #[unroll]
                        for i in 0usize..2 {
                            let mut value = Vector::<e4m3, Const<4>>::empty();
                            #[unroll]
                            for j in 0usize..4 {
                                let (r, c) = def.position_of_nth(lane, (i * 4 + j) as u32, cmma::MatrixIdent::B);
                                value[j] = e4m3::reinterpret(b[(half * 32 + warp % 2 * 16 + column * 8 + c as usize) * 128 + step * 32 + r as usize]);
                            }
                            right[i] = value;
                        }
                        let mut partial = Array::<Vector<f32, Const<2>>>::new(2usize);
                        #[unroll] for i in 0usize..2 { partial[i] = acc[(half * 2 + column) * 2 + i]; }
                        def.execute_inplace(&left, &right, &mut partial);
                        #[unroll] for i in 0usize..2 { acc[(half * 2 + column) * 2 + i] = partial[i]; }
                    }
                }
            }
            #[unroll]
            for half in 0usize..2 {
                #[unroll]
                for column in 0usize..2 {
                    #[unroll]
                    for i in 0usize..2 {
                        let index = (half * 2 + column) * 2 + i;
                        let partial = acc[index];
                        let mut total = sum[index];
                        #[unroll]
                        for j in 0usize..2 {
                            let (r, c) = def.position_of_nth(lane, (i * 2 + j) as u32, cmma::MatrixIdent::Accumulator);
                            let row = row_base + warp / 2 * 16 + r as usize;
                            let col = col_base + warp % 2 * 16 + column * 8 + c as usize + half * width;
                            if row < count {
                                let token = rows[expert * tokens as usize + row] as usize;
                                let sa = input_scale[group * tokens as usize + token];
                                let sb = weight_scale[(expert * (2 * width / 128) + col / 128) * (inner / 128) + group];
                                total[j] += partial[j] * sa * sb;
                            }
                        }
                        sum[index] = total;
                    }
                }
            }
            sync_ruda();
        }
        #[unroll]
        for column in 0usize..2 {
            #[unroll]
            for i in 0usize..2 {
                let up = sum[column * 2 + i];
                let gate = sum[4 + column * 2 + i];
                #[unroll]
                for j in 0usize..2 {
                    let (r, c) = def.position_of_nth(lane, (i * 2 + j) as u32, cmma::MatrixIdent::Accumulator);
                    let row = row_base + warp / 2 * 16 + r as usize;
                    let col = col_base + warp % 2 * 16 + column * 8 + c as usize;
                    if row < count {
                        output[(offset + row) * width + col] = (gate[j] / (1.0f32 + (-gate[j]).exp())) * up[j];
                    }
                }
            }
        }
        tile += RUDA_COUNT_X as usize;
    }
}
