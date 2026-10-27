use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, prelude::*};
use crate::dense_attention::flash_exp2;

#[ruda]
pub(super) fn stage<F: Float>(input: &Array<F>, tile: &mut SharedMemory<F>, row: usize,
    #[comptime] rows: usize, #[comptime] vectorized: bool) {
    let thread = UNIT_POS as usize;
    if comptime!(vectorized) {
        let source = input.to_slice().into_vectorized().with_vector_size::<Const<8>>();
        let mut target = tile.to_slice_mut().into_vectorized().with_vector_size::<Const<8>>();
        for block in 0usize..rows / 16 {
            let r = block * 16 + thread / 16;
            target[r * 17 + thread % 16] = source[(row + r) * 16 + thread % 16];
        }
    } else {
        for block in 0usize..rows / 2 {
            let r = block * 2 + thread / 128;
            tile[r * 136 + thread % 128] = input[(row + r) * 128 + thread % 128];
        }
    }
}

#[ruda]
pub(super) fn dot_columns<F: Float>(a: &SharedMemory<F>, b: &SharedMemory<F>,
    #[comptime] columns: usize) -> Sequence<Vector<f32,Const<4>>> {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut acc = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..2 {
        let mut fragment = Vector::<f32,Const<4>>::empty();
        #[unroll]
        for e in 0usize..4 { fragment[e] = 0.0f32; }
        acc.push(fragment);
    }
    #[unroll]
    for chunk in 0usize..8 {
        let a_start = (warp*16+lane%16)*136+chunk*16+lane/16*8;
        let ar = def.load_matrix_vector::<_,Const<8>>(&a.slice(a_start,a_start+8),4usize,false);
        #[unroll]
        for part in 0usize..2 {
            let b_start = (columns*16+part*8+lane%8)*136+chunk*16+lane/8%2*8;
            let br = def.load_matrix_vector::<_,Const<4>>(&b.slice(b_start,b_start+8),2usize,false);
            def.execute_vectors_inplace(&ar,&br,acc.index_mut(part));
        }
    }
    acc
}

#[ruda]
pub(super) fn dot_pair<F: Float>(a: &SharedMemory<F>, b: &SharedMemory<F>,
    c: &SharedMemory<F>, d: &SharedMemory<F>, #[comptime] columns: usize)
    -> Sequence<Vector<f32,Const<4>>> {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut acc = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..4 { acc.push(Vector::new(0.0f32)); }
    #[unroll]
    for chunk in 0usize..8 {
        let at = (warp*16+lane%16)*136+chunk*16+lane/16*8;
        let ar = def.load_matrix_vector::<_,Const<8>>(&a.slice(at,at+8),4usize,false);
        let cr = def.load_matrix_vector::<_,Const<8>>(&c.slice(at,at+8),4usize,false);
        #[unroll]
        for part in 0usize..2 {
            let bt = (columns*16+part*8+lane%8)*136+chunk*16+lane/8%2*8;
            let br = def.load_matrix_vector::<_,Const<4>>(&b.slice(bt,bt+8),2usize,false);
            let dr = def.load_matrix_vector::<_,Const<4>>(&d.slice(bt,bt+8),2usize,false);
            def.execute_vectors_inplace(&ar,&br,acc.index_mut(part));
            def.execute_vectors_inplace(&cr,&dr,acc.index_mut(2+part));
        }
    }
    acc
}

#[ruda(launch_unchecked)]
pub fn query_registers<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dq: &mut Array<F>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize * 128;
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let head = row / 512 % heads as usize;
    let batch = row / (512 * heads as usize);
    let key = (batch * kv_heads as usize + head / (heads / kv_heads) as usize) * 512;
    let mut q_tile = SharedMemory::<F>::new_aligned(128usize*136,16usize);
    let mut g_tile = SharedMemory::<F>::new_aligned(128usize*136,16usize);
    let mut k_tile = SharedMemory::<F>::new_aligned(64usize*136,16usize);
    let mut v_tile = SharedMemory::<F>::new_aligned(64usize*136,16usize);
    stage(q,&mut q_tile,row,128usize,vectorized);
    stage(grad,&mut g_tile,row,128usize,vectorized);
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut acc = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..16 {
        let mut fragment = Vector::<f32,Const<4>>::empty();
        #[unroll]
        for e in 0usize..4 { fragment[e] = 0.0f32; }
        acc.push(fragment);
    }
    for block in 0usize..8 {
        stage(k,&mut k_tile,key+block*64,64usize,vectorized);
        stage(v,&mut v_tile,key+block*64,64usize,vectorized);
        sync_ruda();
        #[unroll]
        for chunk in 0usize..4 {
            let scores = dot_columns(&q_tile,&k_tile,chunk);
            let derivative = dot_columns(&g_tile,&v_tile,chunk);
            let mut ds = Vector::<F,Const<8>>::empty();
            #[unroll]
            for part in 0usize..2 {
                #[unroll]
                for e in 0usize..4 {
                    let qr = row+warp*16+lane/4+e/2*8;
                    let kr = block*64+chunk*16+part*8+lane%4*2+e%2;
                    let mut allowed = true;
                    if comptime!(causal) { allowed = kr <= qr%512; }
                    let mut value = 0.0f32;
                    if allowed {
                        let p = flash_exp2(scores[part][e]*(scale*1.4426950408889634f32)
                            -stats[qr*2]*1.4426950408889634f32);
                        value = p*(derivative[part][e]-delta[qr]);
                    }
                    ds[part*4+e] = F::cast_from(value);
                }
            }
            #[unroll]
            for j in 0usize..16 {
                let at = (chunk*16+lane%16)*136+j*8;
                let br = def.load_matrix_vector::<_,Const<4>>(&k_tile.slice(at,at+8),2usize,true);
                def.execute_vectors_inplace(&ds,&br,acc.index_mut(j));
            }
        }
        sync_ruda();
    }
    #[unroll]
    for j in 0usize..16 {
        #[unroll]
        for e in 0usize..4 {
            let at = (row+warp*16+lane/4+e/2*8)*128+j*8+lane%4*2+e%2;
            dq[at] = F::cast_from(acc[j][e]*scale);
        }
    }
}

#[ruda(launch_unchecked)]
pub fn key_value_registers<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dk: &mut Array<F>, dv: &mut Array<F>,
    heads: u32, kv_heads: u32, scale: f32, #[comptime] causal: bool,
    #[comptime] vectorized: bool, #[define(F)] _dtype: StorageType) {
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
    stage(k,&mut k_tile,key,128usize,vectorized);
    stage(v,&mut v_tile,key,128usize,vectorized);
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut acc_k = Sequence::<Vector<f32,Const<4>>>::new();
    let mut acc_v = Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll]
    for _ in 0usize..16 {
        let mut a = Vector::<f32,Const<4>>::empty();
        let mut b = Vector::<f32,Const<4>>::empty();
        #[unroll]
        for e in 0usize..4 { a[e] = 0.0f32; b[e] = 0.0f32; }
        acc_k.push(a);
        acc_v.push(b);
    }
    for step in 0usize..8 {
        let block = 7usize-step;
        stage(q,&mut q_tile,query+block*64,64usize,vectorized);
        stage(grad,&mut g_tile,query+block*64,64usize,vectorized);
        sync_ruda();
        #[unroll]
        for chunk in 0usize..4 {
            let scores = dot_columns(&k_tile,&q_tile,chunk);
            let derivative = dot_columns(&v_tile,&g_tile,chunk);
            let mut ds = Vector::<F,Const<8>>::empty();
            let mut probability = Vector::<F,Const<8>>::empty();
            #[unroll]
            for part in 0usize..2 {
                #[unroll]
                for e in 0usize..4 {
                    let kr = key+warp*16+lane/4+e/2*8;
                    let qr = query+block*64+chunk*16+part*8+lane%4*2+e%2;
                    let mut allowed = true;
                    if comptime!(causal) { allowed = kr%512 <= qr%512; }
                    let mut p = 0.0f32;
                    let mut value = 0.0f32;
                    if allowed {
                        p = flash_exp2(scores[part][e]*(scale*1.4426950408889634f32)
                            -stats[qr*2]*1.4426950408889634f32);
                        value = p*(derivative[part][e]-delta[qr]);
                    }
                    ds[part*4+e] = F::cast_from(value);
                    probability[part*4+e] = F::cast_from(p);
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
