use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};
use super::{tile_expert, tile_table};

#[ruda(launch)]
pub fn swiglu(
    input: &Array<u8>, input_scale: &Array<f32>, weights: &Array<u8>, weight_scale: &Array<f32>,
    rows: &Array<u32>, counts: &Array<u32>, offsets: &Array<u32>, output: &mut Array<f32>,
    tokens: u32, #[comptime] inner: usize, #[comptime] width: usize,
) {
    let thread = UNIT_POS as usize;
    let mut table = SharedMemory::<u32>::new(65usize);
    let mut allocation = SharedMemory::<u32>::new(1usize);
    let mut a = SharedMemory::<u8>::new_aligned(128usize * 128, 1024usize);
    let mut b = SharedMemory::<u8>::new_aligned(64usize * 128, 1024usize);
    let mut transpose = SharedMemory::<f32>::new_aligned(128usize * 33, 128usize);
    let complete = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    tile_table(counts, &mut table, 128usize);
    if thread < 32 {
        tcgen05::allocate(&mut allocation.slice_mut(0usize, 1usize), 64u32);
        tcgen05::relinquish();
    }
    sync_ruda();
    let address = allocation[0usize];
    let m_tiles = table[32usize] as usize;
    let mut task = RUDA_POS_X as usize;
    let mut phase = 0u32;
    while task < m_tiles * (width / 32) {
        let m_tile = task % m_tiles;
        let expert = tile_expert(&table, m_tile);
        let row_base = (m_tile - table[expert] as usize) * 128;
        let col_base = task / m_tiles * 32;
        let count = table[33usize + expert] as usize;
        let offset = offsets[expert] as usize;
        let row = row_base + thread;
        let mut token = 0usize;
        if row < count { token = rows[expert * tokens as usize + row] as usize; }
        let mut sum = Array::<f32>::new(64usize);
        #[unroll] for i in 0usize..64 { sum[i] = 0.0; }
        for group in 0usize..(inner / 128) {
            #[unroll]
            for load in 0usize..32 {
                let p = (thread + load * 128) * 4;
                let r = row_base + p / 128;
                let mut data = Vector::<u8, Const<4>>::new(0u8);
                if r < count {
                    let source_row = rows[expert * tokens as usize + r] as usize;
                    let packed = input.to_slice().into_vectorized().with_vector_size::<Const<4>>();
                    data = packed[(source_row * inner + group * 128 + p % 128) / 4];
                }
                let swizzled = p ^ ((p / 128 % 8) * 16);
                #[unroll] for j in 0usize..4 { a[swizzled + j] = data[j]; }
            }
            #[unroll]
            for load in 0usize..16 {
                let p = (thread + load * 128) * 4;
                let col = col_base + p / 128 % 32 + p / (32 * 128) * width;
                let packed = weights.to_slice().into_vectorized().with_vector_size::<Const<4>>();
                let data = packed[((expert * width * 2 + col) * inner + group * 128 + p % 128) / 4];
                let swizzled = p ^ ((p / 128 % 8) * 16);
                #[unroll] for j in 0usize..4 { b[swizzled + j] = data[j]; }
            }
            sync_async_proxy_shared();
            sync_ruda();
            if thread == 0 {
                tcgen05::fence(false);
                #[unroll]
                for step in 0usize..4 {
                    let ad = tcgen05::descriptor(&a.slice(step * 32, step * 32 + 16));
                    let bd = tcgen05::descriptor(&b.slice(step * 32, step * 32 + 16));
                    tcgen05::mma_fp8(address, ad, bd, step != 0, 64u32, false, false);
                }
                tcgen05::commit(&complete);
                complete.wait_parity(phase);
                tcgen05::fence(true);
            }
            sync_ruda();
            tcgen05::fence(false);
            let mut sa = 0.0f32;
            if row < count { sa = input_scale[group * tokens as usize + token]; }
            #[unroll]
            for half in 0usize..2 {
                let partial = tcgen05::load(address + ((UNIT_POS / 32 * 32) << 16) + (half * 32) as u32);
                let col = col_base + half * width;
                let sb = weight_scale[(expert * (width * 2 / 128) + col / 128) * (inner / 128) + group];
                #[unroll] for j in 0usize..32 { sum[half * 32 + j] += partial[j] * sa * sb; }
            }
            tcgen05::fence(true);
            sync_ruda();
            phase ^= 1u32;
        }
        #[unroll]
        for col in 0usize..32 {
            let gate = sum[32 + col];
            transpose[thread * 33 + col] = (gate / (1.0f32 + (-gate).exp())) * sum[col];
        }
        sync_ruda();
        #[unroll]
        for copy in 0usize..32 {
            let p = thread + copy * 128;
            let r = row_base + p / 32;
            if r < count { output[(offset + r) * width + col_base + p % 32] = transpose[p / 32 * 33 + p % 32]; }
        }
        sync_ruda();
        task += RUDA_COUNT_X as usize;
    }
    tcgen05::fence(true);
    sync_ruda();
    if thread < 32 { tcgen05::deallocate(address, 64u32); }
}
