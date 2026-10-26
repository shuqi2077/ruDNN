use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, prelude::*};
use crate::dense_attention::flash_exp2;
use super::register_backward::{dot_pair, stage};

#[ruda]
fn masked_operands_finite<F: Float>(q: &SharedMemory<F>, k: &SharedMemory<F>,
    grad: &SharedMemory<F>, workspace: &mut SharedMemory<F>) -> bool {
    let thread = UNIT_POS as usize;
    let mut finite = true;
    for chunk in 0usize..32 {
        let pos = chunk*256+thread;
        let at = pos/128*136+pos%128;
        finite = finite && f32::abs(f32::cast_from(q[at])) < f32::INFINITY
            && f32::abs(f32::cast_from(grad[at])) < f32::INFINITY
            && f32::abs(f32::cast_from(k[at])) < f32::INFINITY
            && f32::abs(f32::cast_from(k[at+64*136])) < f32::INFINITY;
    }
    let warp_finite = plane_all(finite);
    if thread%32 == 0 { workspace[thread/32] = F::cast_from(u32::cast_from(warp_finite)); }
    sync_ruda();
    let mut all_finite = true;
    #[unroll]
    for warp in 0usize..8 { all_finite = all_finite && workspace[warp] != F::new(0.0); }
    all_finite
}

#[ruda(launch_unchecked)]
pub fn fused_key_value<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, partial_q: &mut Array<f32>,
    dk: &mut Array<F>, dv: &mut Array<F>, heads: u32, kv_heads: u32, scale: f32,
    #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    let key = RUDA_POS_X as usize * 128;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let kh = key / 512 % kv_heads as usize;
    let batch = key / (512 * kv_heads as usize);
    let group = (heads / kv_heads) as usize;
    let query = (batch*heads as usize+kh*group+RUDA_POS_Y as usize)*512;
    let mut k_tile = SharedMemory::<F>::new_aligned(128usize*136,16usize);
    let mut v_tile = SharedMemory::<F>::new_aligned(128usize*136,16usize);
    let mut q_tile = SharedMemory::<F>::new_aligned(64usize*136,16usize);
    let mut g_tile = SharedMemory::<F>::new_aligned(64usize*136,16usize);
    let mut ds_tile = SharedMemory::<F>::new_aligned(128usize*72,16usize);
    stage(k,&mut k_tile,key,128usize,vectorized);
    stage(v,&mut v_tile,key,128usize,vectorized);
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut acc_k = Sequence::<Vector<f32,Const<4>>>::new();
    let mut acc_v = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..16 {
        acc_k.push(Vector::new(0.0f32));
        acc_v.push(Vector::new(0.0f32));
    }
    for block in 0usize..8 {
        stage(q,&mut q_tile,query+block*64,64usize,vectorized);
        stage(grad,&mut g_tile,query+block*64,64usize,vectorized);
        sync_ruda();
        let mut skip = false;
        if comptime!(causal) {
            if block*64+63 < key%512 {
                skip = masked_operands_finite(&q_tile,&k_tile,&g_tile,&mut ds_tile);
                sync_ruda();
            }
        }
        if skip {
            for chunk in 0usize..32 {
                partial_q[(key%512/128)*q.len()+(query+block*64)*128+chunk*256+UNIT_POS as usize] = 0.0f32;
            }
        } else {
        #[unroll]
        for chunk in 0usize..4 {
            let products = dot_pair(&k_tile,&q_tile,&v_tile,&g_tile,chunk);
            let mut ds = Vector::<F,Const<8>>::empty();
            let mut probability = Vector::<F,Const<8>>::empty();
            #[unroll]
            for part in 0usize..2 {
                #[unroll]
                for e in 0usize..4 {
                    let kr = key+warp*16+lane/4+e/2*8;
                    let qc = chunk*16+part*8+lane%4*2+e%2;
                    let qr = query+block*64+qc;
                    let mut allowed = true;
                    if comptime!(causal) { allowed = kr%512 <= qr%512; }
                    let mut p = 0.0f32;
                    let mut value = 0.0f32;
                    if allowed {
                        p = flash_exp2(products[part][e]*(scale*1.4426950408889634f32)
                            -stats[qr*2]*1.4426950408889634f32);
                        value = p*(products[2+part][e]-delta[qr]);
                    }
                    ds[part*4+e] = F::cast_from(value);
                    probability[part*4+e] = F::cast_from(p);
                    ds_tile[(warp*16+lane/4+e/2*8)*72+qc] = ds[part*4+e];
                }
            }
            #[unroll]
            for j in 0usize..16 {
                let at = (chunk*16+lane%16)*136+j*8;
                let qr = def.load_matrix_vector::<_,Const<4>>(&q_tile.slice(at,at+8),2usize,true);
                let gr = def.load_matrix_vector::<_,Const<4>>(&g_tile.slice(at,at+8),2usize,true);
                def.execute_vectors_inplace(&ds,&qr,acc_k.index_mut(j));
                def.execute_vectors_inplace(&probability,&gr,acc_v.index_mut(j));
            }
        }
        sync_ruda();
        for j in 0usize..8 {
            let mut acc = Vector::<f32,Const<4>>::new(0.0f32);
            #[unroll]
            for chunk in 0usize..8 {
                let at = (chunk*16+lane%8+lane/16*8)*72+warp%4*16+lane/8%2*8;
                let ar = def.load_matrix_vector::<_,Const<8>>(&ds_tile.slice(at,at+8),4usize,true);
                let bt = (chunk*16+lane%16)*136+warp/4*64+j*8;
                let br = def.load_matrix_vector::<_,Const<4>>(&k_tile.slice(bt,bt+8),2usize,true);
                def.execute_vectors_inplace(&ar,&br,&mut acc);
            }
            #[unroll]
            for e in 0usize..4 {
                let row = query+block*64+warp%4*16+lane/4+e/2*8;
                let col = warp/4*64+j*8+lane%4*2+e%2;
                partial_q[(key%512/128)*q.len()+row*128+col] = acc[e];
            }
        }
        }
        sync_ruda();
    }
    #[unroll]
    for j in 0usize..16 {
        #[unroll]
        for e in 0usize..4 {
            let at = (key+warp*16+lane/4+e/2*8)*128+j*8+lane%4*2+e%2
                +RUDA_POS_Y as usize*(dk.len()/group);
            dk[at] = F::cast_from(0.0f32+f32::cast_from(F::cast_from(acc_k[j][e]*scale)));
            dv[at] = F::cast_from(0.0f32+f32::cast_from(F::cast_from(acc_v[j][e])));
        }
    }
}

#[ruda(launch_unchecked)]
pub fn reduce_query<F: Float>(partial: &Array<f32>, dq: &mut Array<F>, scale: f32,
    #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < dq.len() {
        let mut sum = partial[pos];
        #[unroll]
        for block in 1usize..4 { sum += partial[block*dq.len()+pos]; }
        dq[pos] = F::cast_from(sum*scale);
    }
}
