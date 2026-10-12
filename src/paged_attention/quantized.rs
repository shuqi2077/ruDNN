//! Paged attention reads affine packed KV values directly into registers.
use super::*;
use ruda_kernel::quantization::packed::{PackedTensor,load};
use ruda_kernel::dsl as kernel_dsl;

impl<R:Runtime> DevicePlan<R> {
    /// Inference over quantized MHA/GQA caches, or quantized latent/position MLA.
    /// K/V logical shapes remain [pages,page_size,KV_heads,features].
    pub fn attention_quantized(&self,q:&RudaTensor<R>,k:&PackedTensor<R>,v:&PackedTensor<R>,
        position:Option<(&RudaTensor<R>,&PackedTensor<R>)>,scale:f32,causal:bool,
        workspace:Option<&mut SplitWorkspace<R>>)->Result<RudaTensor<R>,PagedAttentionError> {
        check_tensor(q,q)?;
        if q.meta.num_dims()!=3 || k.shape.len()!=4 || v.shape.len()!=4 {
            return Err(PagedAttentionError("quantized paged query/cache ranks must be 3/4/4"));
        }
        for cache in [k,v] { cache.validate(); cache.check(q); }
        let h=q.meta.shape()[1];let d=k.width();let dv=v.width();let kh=k.shape[2];
        if h==0 || kh==0 || h%kh!=0 || d>1024 || dv>1024 || q.meta.shape()[2]!=d
            || q.meta.shape()[0]!=self.host.queries || k.shape[..3]!=v.shape[..3]
            || k.shape[0]!=self.host.pages || k.shape[1]!=self.host.page_size
            || k.group_index.meta.num_elements()!=0 || v.group_index.meta.num_elements()!=0
            || !scale.is_finite() || scale<=0.0 {
            return Err(PagedAttentionError("invalid quantized paged cache geometry/scale"));
        }
        if self.metadata.device.to_id()!=q.device.to_id() || !self.metadata.client.same_execution_queue(&q.client) {
            return Err(PagedAttentionError("quantized metadata belongs to another device/queue"));
        }
        let (qp,kp,p)=if let Some((qp,kp))=position {
            check_tensor(qp,q)?;kp.validate();kp.check(q);
            if kh!=1 || d!=dv || qp.meta.num_dims()!=3 || kp.shape.len()!=4
                || qp.meta.shape()[..2]!=q.meta.shape()[..2] || kp.shape[..3]!=k.shape[..3]
                || qp.meta.shape()[2]!=kp.width() || kp.width()>256 || kp.group_index.meta.num_elements()!=0 {
                return Err(PagedAttentionError("invalid quantized MLA positional geometry"));
            }
            (qp,kp,kp.width())
        } else {(q,k,0)};
        let lanes=q.client.properties().hardware.plane_size_max;
        let max=q.client.properties().hardware.max_ruda_count;
        if !matches!(lanes,32|64) || self.host.queries>max.0 as usize || h>max.1 as usize {
            return Err(PagedAttentionError("quantized paged attention requires a full plane and legal launch grid"));
        }
        let shape=Shape::new([self.host.queries,h,dv]);bounded(&shape)?;
        let out=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),shape,q.dtype);
        let (splits,target,dtype)=if let Some(w)=workspace {
            if !w.matches(q,self.host.queries,h,dv,w.splits()) || w.splits()>max.2 as usize {
                return Err(PagedAttentionError("quantized split workspace mismatch"));
            }
            (w.splits(),&w.tensor,DType::F32)
        } else {(1,&out,q.dtype)};
        if self.host.queries==0 { return Ok(out); }
        quantized_attention::launch::<R>(&q.client,
            RudaCount::Static(self.host.queries as u32,h as u32,splits as u32),RudaDim::new_1d(lanes),
            q.clone().into_array_arg(),qp.clone().into_array_arg(),
            k.codes.clone().into_array_arg(),k.scales.clone().into_array_arg(),k.offsets.clone().into_array_arg(),
            v.codes.clone().into_array_arg(),v.scales.clone().into_array_arg(),v.offsets.clone().into_array_arg(),
            kp.codes.clone().into_array_arg(),kp.scales.clone().into_array_arg(),kp.offsets.clone().into_array_arg(),
            k.group_index.clone().into_array_arg(),self.metadata.clone().into_array_arg(),target.clone().into_array_arg(),
            scale,self.host.queries as u32,self.host.sequences as u32,self.host.table_width as u32,
            self.host.page_size as u32,h as u32,kh as u32,d,dv,p,k.group_size,v.group_size,kp.group_size,
            k.format.code(),v.format.code(),kp.format.code(),lanes as usize,causal,splits,q.dtype.into(),dtype.into());
        if splits>1 {
            kernel::merge::launch::<R>(&q.client,RudaCount::Static(self.host.queries as u32,h as u32,1),RudaDim::new_1d(lanes),
                target.clone().into_array_arg(),out.clone().into_array_arg(),h as u32,dv,lanes as usize,splits,q.dtype.into());
        }
        Ok(out)
    }
}

#[ruda(launch)]
fn quantized_attention<F:Float,O:Float>(q:&Array<F>,qp:&Array<F>,
    k:&Array<u8>,ks:&Array<f32>,ko:&Array<f32>,v:&Array<u8>,vs:&Array<f32>,vo:&Array<f32>,
    kp:&Array<u8>,ps:&Array<f32>,po:&Array<f32>,empty_index:&Array<i32>,metadata:&Array<u32>,out:&mut Array<O>,
    scale:f32,queries:u32,sequences:u32,table_width:u32,page_size:u32,heads:u32,kv_heads:u32,
    #[comptime] key_dim:usize,#[comptime] value_dim:usize,#[comptime] position_dim:usize,
    #[comptime] kg:usize,#[comptime] vg:usize,#[comptime] pg:usize,
    #[comptime] kf:u32,#[comptime] vf:u32,#[comptime] pf:u32,
    #[comptime] lanes:usize,#[comptime] causal:bool,#[comptime] splits:usize,
    #[define(F)] _dtype:StorageType,#[define(O)] _output_dtype:StorageType) {
    let row=RUDA_POS_X as usize;let head=RUDA_POS_Y as usize;let lane=UNIT_POS as usize;
    let sequence=metadata[row] as usize;let position=metadata[queries as usize+row] as usize;
    let mut end=metadata[2*queries as usize+sequence] as usize;
    if comptime!(causal) { if position<end {end=position+1;} }
    let split=RUDA_POS_Z as usize;let base=end/splits;let extra=end%splits;
    let begin=split*base+usize::min(split,extra);
    let mut span=base;if split<extra {span+=1;} end=begin+span;
    let kv_head=head/(heads/kv_heads) as usize;
    let mut query=Array::<f32>::new((key_dim+lanes-1)/lanes);
    let mut query_pos=Array::<f32>::new((position_dim+lanes)/lanes);
    let mut acc=Array::<f32>::new((value_dim+lanes-1)/lanes);
    #[unroll]
    for j in 0usize..((key_dim+lanes-1)/lanes) {
        let d=lane+j*lanes;query[j]=0.0;
        if d<key_dim {query[j]=f32::cast_from(q[(row*heads as usize+head)*key_dim+d]);}
    }
    if comptime!(position_dim>0) {
        #[unroll]
        for j in 0usize..((position_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;query_pos[j]=0.0;
            if d<position_dim {query_pos[j]=f32::cast_from(qp[(row*heads as usize+head)*position_dim+d]);}
        }
    }
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) {acc[j]=0.0;}
    let mut maximum=0.0f32;let mut denominator=0.0f32;
    for token in begin..end {
        let logical=token/page_size as usize;let offset=token%page_size as usize;
        let page=metadata[2*queries as usize+sequences as usize+sequence*table_width as usize+logical] as usize;
        let cr=(page*page_size as usize+offset)*kv_heads as usize+kv_head;
        let mut partial=0.0f32;
        #[unroll]
        for j in 0usize..((key_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            if d<key_dim {partial=fma(query[j],load(k,ks,ko,empty_index,cr,d,key_dim,kg,kf,false),partial);}
        }
        if comptime!(position_dim>0) {
            #[unroll]
            for j in 0usize..((position_dim+lanes-1)/lanes) {
                let d=lane+j*lanes;
                if d<position_dim {partial=fma(query_pos[j],load(kp,ps,po,empty_index,cr,d,position_dim,pg,pf,false),partial);}
            }
        }
        let score=plane_sum(partial)*scale;
        let mut next=score;if token!=begin {next=f32::max(maximum,score);}
        let mut previous=0.0f32;if token!=begin {previous=(maximum-next).exp();}
        let weight=(score-next).exp();denominator=fma(denominator,previous,weight);
        #[unroll]
        for j in 0usize..((value_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            if d<value_dim {acc[j]=fma(weight,load(v,vs,vo,empty_index,cr,d,value_dim,vg,vf,false),acc[j]*previous);}
        }
        maximum=next;
    }
    let output_row=row*heads as usize+head;
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) {
        let d=lane+j*lanes;
        if d<value_dim {
            if comptime!(splits==1) {
                let mut result=0.0f32;if span!=0 {result=acc[j]/denominator;}
                out[output_row*value_dim+d]=O::cast_from(result);
            } else {out[(output_row*splits+split)*(value_dim+2)+d]=O::cast_from(acc[j]);}
        }
    }
    if comptime!(splits>1) {if lane==0 {
        let at=(output_row*splits+split)*(value_dim+2)+value_dim;
        out[at]=O::cast_from(maximum);out[at+1]=O::cast_from(denominator);
    }}
}
