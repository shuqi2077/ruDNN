//! Split-KV dense decode with FP32 partials and a stable softmax merge.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub fn partial_parallel<F: Float>(
    q: &Array<F>, k: &Array<F>, v: &Array<F>,
    values: &mut Array<f32>, statistics: &mut Array<f32>,
    heads: u32, kv_heads: u32, keys: u32, scale: f32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] splits: usize, #[define(F)] _dtype: StorageType,
) {
    let part_row=RUDA_POS_X as usize;
    let row=part_row/splits;let split=part_row%splits;
    let unit=UNIT_POS as usize;let lane=unit%32;let warp=unit/32;
    let group=unit/8;let component=unit%8;
    let head=row%heads as usize;let batch=row/heads as usize;
    let kh=head/(heads/kv_heads) as usize;
    let span=(keys as usize+splits-1)/splits;
    let begin=split*span;let end=usize::min(begin+span,keys as usize);
    let mut scores=SharedMemory::<f32>::new(128usize);
    let mut reductions=SharedMemory::<f32>::new(4usize);
    let mut query=Array::<f32>::new((key_dim+7)/8);
    #[unroll] for d in 0usize..((key_dim+7)/8) {
        query[d]=0.0;
        if component+d*8<key_dim {query[d]=f32::cast_from(q[row*key_dim+component+d*8]);}
    }
    scores[unit]=f32::cast_from(f32::NEG_INFINITY);
    sync_ruda();
    for chunk in 0usize..((span+15)/16) {
        let local=chunk*16+group;let token=begin+local;
        let cache_row=(batch*kv_heads as usize+kh)*keys as usize+token;
        let mut dot=0.0f32;
        #[unroll] for d in 0usize..((key_dim+7)/8) {
            if token<end && component+d*8<key_dim {
                dot=fma(query[d],f32::cast_from(k[cache_row*key_dim+component+d*8]),dot);
            }
        }
        dot+=plane_shuffle_xor(dot,4u32);
        dot+=plane_shuffle_xor(dot,2u32);
        dot+=plane_shuffle_xor(dot,1u32);
        if component==0 && token<end {scores[local]=dot*scale;}
    }
    sync_ruda();
    let score=scores[unit];let plane_maximum=plane_max(score);
    if lane==0 {reductions[warp]=plane_maximum;}
    sync_ruda();
    let mut maximum=f32::cast_from(f32::NEG_INFINITY);
    #[unroll] for i in 0usize..4 {maximum=f32::max(maximum,reductions[i]);}
    let mut probability=0.0f32;
    if unit<span && begin+unit<end {probability=(score-maximum).exp();}
    scores[unit]=probability;
    let plane_denominator=plane_sum(probability);
    sync_ruda();
    if lane==0 {reductions[warp]=plane_denominator;}
    sync_ruda();
    #[unroll] for j in 0usize..((value_dim+127)/128) {
        let d=unit+j*128;
        if d<value_dim {
            let mut total=0.0f32;
            for token in begin..end {
                let cache_row=(batch*kv_heads as usize+kh)*keys as usize+token;
                total=fma(scores[token-begin],f32::cast_from(v[cache_row*value_dim+d]),total);
            }
            values[part_row*value_dim+d]=total;
        }
    }
    if unit==0 {
        let mut denominator=0.0f32;
        #[unroll] for i in 0usize..4 {denominator+=reductions[i];}
        statistics[part_row*2]=maximum;statistics[part_row*2+1]=denominator;
    }
}

#[ruda(launch)]
pub fn merge_parallel<F: Float>(
    values: &Array<f32>, statistics: &Array<f32>,
    output: &mut Array<F>, final_statistics: &mut Array<f32>,
    #[comptime] value_dim: usize, #[comptime] splits: usize,
    #[define(F)] _dtype: StorageType,
) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let mut part_maximum=f32::cast_from(f32::NEG_INFINITY);
    let mut part_denominator=0.0f32;
    if lane<splits {
        part_denominator=statistics[(row*splits+lane)*2+1];
        if part_denominator!=0.0f32 {part_maximum=statistics[(row*splits+lane)*2];}
    }
    let maximum=plane_max(part_maximum);
    let mut scale=0.0f32;
    if part_denominator!=0.0f32 {scale=(part_maximum-maximum).exp();}
    let denominator=plane_sum(part_denominator*scale);
    let mut sums=Array::<f32>::new((value_dim+31)/32);
    #[unroll] for j in 0usize..((value_dim+31)/32) {sums[j]=0.0;}
    #[unroll] for split in 0usize..splits {
        let factor=plane_shuffle(scale,split as u32);
        #[unroll] for j in 0usize..((value_dim+31)/32) {
            let d=lane+j*32;
            if d<value_dim {sums[j]+=values[(row*splits+split)*value_dim+d]*factor;}
        }
    }
    #[unroll] for j in 0usize..((value_dim+31)/32) {
        let d=lane+j*32;
        if d<value_dim {
            let mut value=sums[j];if denominator!=0.0f32 {value/=denominator;}
            output[row*value_dim+d]=F::cast_from(value);
        }
    }
    if lane==0 {final_statistics[row*2]=maximum;final_statistics[row*2+1]=denominator;}
}

#[ruda(launch)]
pub fn partial<F: Float>(
    q: &Array<F>, k: &Array<F>, v: &Array<F>,
    values: &mut Array<f32>, statistics: &mut Array<f32>,
    heads: u32, kv_heads: u32, keys: u32, scale: f32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] splits: usize, #[define(F)] _dtype: StorageType,
) {
    let part_row = RUDA_POS_X as usize;
    let row = part_row / splits;
    let split = part_row % splits;
    let lane = UNIT_POS as usize;
    let head = row % heads as usize;
    let batch = row / heads as usize;
    let kh = head / (heads / kv_heads) as usize;
    let mut query = Array::<f32>::new((key_dim + 31) / 32);
    let mut accumulator = Array::<f32>::new((value_dim + 31) / 32);
    #[unroll]
    for j in 0usize..((key_dim + 31) / 32) {
        let d = lane + j * 32;
        query[j] = 0.0;
        if d < key_dim { query[j] = f32::cast_from(q[row * key_dim + d]); }
    }
    #[unroll]
    for j in 0usize..((value_dim + 31) / 32) { accumulator[j] = 0.0; }
    let span = (keys as usize + splits - 1) / splits;
    let begin = split * span;
    let end = usize::min(begin + span, keys as usize);
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    for token in begin..end {
        let cache_row = (batch * kv_heads as usize + kh) * keys as usize + token;
        let mut dot = 0.0f32;
        #[unroll]
        for j in 0usize..((key_dim + 31) / 32) {
            let d = lane + j * 32;
            if d < key_dim { dot = fma(query[j], f32::cast_from(k[cache_row * key_dim + d]), dot); }
        }
        let score = plane_sum(dot) * scale;
        let mut next_maximum = score;
        let mut old_scale = 0.0f32;
        if denominator != 0.0 {
            next_maximum = f32::max(maximum, score);
            old_scale = (maximum - next_maximum).exp();
        }
        let probability = (score - next_maximum).exp();
        denominator = fma(denominator, old_scale, probability);
        maximum = next_maximum;
        #[unroll]
        for j in 0usize..((value_dim + 31) / 32) {
            let d = lane + j * 32;
            if d < value_dim {
                accumulator[j] = fma(probability, f32::cast_from(v[cache_row * value_dim + d]), accumulator[j] * old_scale);
            }
        }
    }
    #[unroll]
    for j in 0usize..((value_dim + 31) / 32) {
        let d = lane + j * 32;
        if d < value_dim { values[part_row * value_dim + d] = accumulator[j]; }
    }
    if lane == 0 {
        statistics[part_row * 2] = maximum;
        statistics[part_row * 2 + 1] = denominator;
    }
}

#[ruda(launch)]
pub fn merge<F: Float>(
    values: &Array<f32>, statistics: &Array<f32>,
    output: &mut Array<F>, final_statistics: &mut Array<f32>,
    #[comptime] value_dim: usize, #[comptime] splits: usize,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut maximum = f32::cast_from(f32::NEG_INFINITY);
    for split in 0usize..splits {
        let index = (row * splits + split) * 2;
        if statistics[index + 1] != 0.0 { maximum = f32::max(maximum, statistics[index]); }
    }
    let mut denominator = 0.0f32;
    for split in 0usize..splits {
        let index = (row * splits + split) * 2;
        if statistics[index + 1] != 0.0 {
            denominator += statistics[index + 1] * (statistics[index] - maximum).exp();
        }
    }
    #[unroll]
    for j in 0usize..((value_dim + 31) / 32) {
        let d = lane + j * 32;
        if d < value_dim {
            let mut result = 0.0f32;
            for split in 0usize..splits {
                let part = row * splits + split;
                if statistics[part * 2 + 1] != 0.0 {
                    result += values[part * value_dim + d] * (statistics[part * 2] - maximum).exp();
                }
            }
            if denominator != 0.0 { result /= denominator; }
            output[row * value_dim + d] = F::cast_from(result);
        }
    }
    if lane == 0 {
        final_statistics[row * 2] = maximum;
        final_statistics[row * 2 + 1] = denominator;
    }
}
