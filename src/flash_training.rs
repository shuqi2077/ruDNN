//! Native SM80 BF16 forward with warp MMA and Flash storage boundaries.
pub mod backward;
pub mod register_backward;
pub mod fused_backward;
#[cfg(test)]
mod tests;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, prelude::*};
use crate::dense_attention::{flash_exp2,flash_log};

#[ruda(launch_unchecked)]
pub fn forward_registers<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    out: &mut Array<F>, stats: &mut Array<f32>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    let row_base = RUDA_POS_X as usize * 128;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let head = (row_base / 512) % heads as usize;
    let batch = row_base / (heads as usize * 512);
    let kv_base = (batch * kv_heads as usize + head / (heads / kv_heads) as usize) * 512;
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut q_tile = SharedMemory::<F>::new_aligned(128usize * 136, 16usize);
    let mut k_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    let mut v_tile = SharedMemory::<F>::new_aligned(64usize * 136, 16usize);
    backward::stage(q, &mut q_tile, row_base, 128usize, vectorized);
    let mut output_acc = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..32 { output_acc.push(Vector::new(0.0f32)); }
    let mut maximum = Vector::<f32,Const<4>>::new(f32::NEG_INFINITY);
    let mut denominator = Vector::<f32,Const<4>>::new(0.0f32);
    let scale_log2 = scale * 1.4426950408889634f32;
    let mut blocks = 8usize;
    if comptime!(causal) { blocks = (row_base % 512 / 128 + 1) * 2; }
    for step in 0usize..blocks {
        let block = blocks - 1 - step;
        backward::stage(k, &mut k_tile, kv_base + block * 64, 64usize, vectorized);
        backward::stage(v, &mut v_tile, kv_base + block * 64, 64usize, vectorized);
        sync_ruda();
        let mut scores = Sequence::<Vector<f32,Const<4>>>::new();
        #[unroll]
        for _ in 0usize..16 { scores.push(Vector::new(0.0f32)); }
        #[unroll]
        for chunk in 0usize..8 {
            let mut ar = Sequence::<Vector<F,Const<8>>>::new();
            #[unroll]
            for i in 0usize..2 {
                let at = (warp * 32 + i * 16 + lane % 16) * 136 + chunk * 16 + lane / 16 * 8;
                ar.push(def.load_matrix_vector::<_,Const<8>>(&q_tile.slice(at,at+8),4usize,false));
            }
            #[unroll]
            for j in 0usize..8 {
                let at = (j * 8 + lane % 8) * 136 + chunk * 16 + lane / 8 % 2 * 8;
                let br = def.load_matrix_vector::<_,Const<4>>(&k_tile.slice(at,at+8),2usize,false);
                #[unroll]
                for i in 0usize..2 { def.execute_vectors_inplace(&ar[i],&br,scores.index_mut(i*8+j)); }
            }
        }
        let mut rescale = Vector::<f32,Const<4>>::empty();
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for r in 0usize..2 {
                let row = row_base + warp * 32 + i * 16 + lane / 4 + r * 8;
                let mut local_max = f32::new(f32::NEG_INFINITY);
                #[unroll]
                for j in 0usize..8 {
                    let fragment = scores.index_mut(i*8+j);
                    #[unroll]
                    for c in 0usize..2 {
                        if comptime!(causal) {
                            if block*64+j*8+lane%4*2+c > row%512 { fragment[r*2+c] = f32::NEG_INFINITY; }
                        }
                        local_max = f32::max(local_max,fragment[r*2+c]);
                    }
                }
                let previous = maximum[i*2+r];
                let mut next = previous;
                #[unroll]
                for p in 0u32..4 { next = f32::max(next,plane_shuffle(local_max,(lane as u32/4)*4+p)); }
                let mut factor = 0.0f32;
                if previous != f32::NEG_INFINITY { factor = flash_exp2((previous-next)*scale_log2); }
                maximum[i*2+r] = next;
                rescale[i*2+r] = factor;
                let mut max_scaled = 0.0f32;
                if next != f32::NEG_INFINITY { max_scaled = next * scale_log2; }
                let mut sum = denominator[i*2+r];
                #[unroll]
                for j in 0usize..8 {
                    let fragment = scores.index_mut(i*8+j);
                    #[unroll]
                    for c in 0usize..2 {
                        let p = flash_exp2(fragment[r*2+c]*scale_log2-max_scaled);
                        if comptime!(j == 0 && c == 0) { sum = fma(sum,factor,p); }
                        else { sum += p; }
                        fragment[r*2+c] = p;
                    }
                }
                denominator[i*2+r] = sum;
            }
        }
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..16 {
                let fragment = output_acc.index_mut(i*16+j);
                #[unroll]
                for r in 0usize..2 {
                    #[unroll]
                    for c in 0usize..2 { fragment[r*2+c] *= rescale[i*2+r]; }
                }
            }
        }
        #[unroll]
        for chunk in 0usize..4 {
            let mut ar = Sequence::<Vector<F,Const<8>>>::new();
            #[unroll]
            for i in 0usize..2 {
                let mut fragment = Vector::<F,Const<8>>::empty();
                #[unroll]
                for part in 0usize..2 {
                    #[unroll]
                    for e in 0usize..4 { fragment[part*4+e] = F::cast_from(scores[i*8+chunk*2+part][e]); }
                }
                ar.push(fragment);
            }
            #[unroll]
            for j in 0usize..16 {
                let at = (chunk*16+lane%16)*136+j*8;
                let br = def.load_matrix_vector::<_,Const<4>>(&v_tile.slice(at,at+8),2usize,true);
                #[unroll]
                for i in 0usize..2 { def.execute_vectors_inplace(&ar[i],&br,output_acc.index_mut(i*16+j)); }
            }
        }
        sync_ruda();
    }
    let mut inverse = Vector::<f32,Const<4>>::empty();
    #[unroll]
    for i in 0usize..2 {
        #[unroll]
        for r in 0usize..2 {
            let d = denominator[i*2+r];
            let base = (lane as u32/4)*4;
            let sum = (plane_shuffle(d,base)+plane_shuffle(d,base+2))
                +(plane_shuffle(d,base+1)+plane_shuffle(d,base+3));
            inverse[i*2+r] = 1.0f32/sum;
            if lane % 4 == 0 {
                let row = row_base+warp*32+i*16+lane/4+r*8;
                stats[row*2] = fma(maximum[i*2+r],scale,flash_log(sum));
                stats[row*2+1] = sum;
            }
        }
    }
    #[unroll]
    for i in 0usize..2 {
        #[unroll]
        for j in 0usize..16 {
            #[unroll]
            for r in 0usize..2 {
                #[unroll]
                for c in 0usize..2 {
                    let index = (row_base+warp*32+i*16+lane/4+r*8)*128+j*8+lane%4*2+c;
                    let value = F::cast_from(output_acc[i*16+j][r*2+c]*inverse[i*2+r]);
                    out[index] = value;
                    stats[out.len()/128*2+index] = f32::cast_from(value);
                }
            }
        }
    }
}
