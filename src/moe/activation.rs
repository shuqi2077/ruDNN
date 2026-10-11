use super::{MoeError,elements,empty,float_tensor,same_device,kernels};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::{RudaTensor,contiguous::into_contiguous}};
use ruda_kernel::tensor::{layout::{address_type, max_vector_size}, tuning};

/// Requested actual native storage-rounded SwiGLU input derivatives.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct SwiGluActivationSelection {pub gate:bool,pub up:bool}
/// Only actual requested original native activation derivatives, without placeholder outputs.
#[derive(Debug)]
pub struct SwiGluActivationBackward<R:Runtime> {pub gate:Option<RudaTensor<R>>,pub up:Option<RudaTensor<R>>}
fn validate<R:Runtime>(gate:&RudaTensor<R>,up:&RudaTensor<R>) -> Result<usize,MoeError> {
    float_tensor(gate)?;float_tensor(up)?;same_device(gate,up)?;
    if gate.meta.shape()!=up.meta.shape() || gate.dtype!=up.dtype || !gate.client.same_execution_queue(&up.client) {
        return Err(MoeError("native SwiGLU requires matching original floating geometry/storage/device/queue"));
    }
    elements(gate.meta.shape())
}
/// Original ruDNN SwiGLU: round stored SiLU before multiplying by up, then round output storage.
/// Source input handles are never overwritten, so actual primals remain available for AD.
pub fn swiglu_activation<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {
    let size=validate(&gate,&up)?;
    if size!=0 && tuning::is_enabled() {
        let maximum=gate.client.properties().hardware.max_ruda_dim.0.min(gate.client.properties().hardware.max_units_per_ruda);
        let mut candidates=vec![("original_launch",(0,false)),("vector_launch",(0,true))];
        for (scalar,vector,units) in [("scalar_128","vector_128",128),("scalar_256","vector_256",256),("scalar_512","vector_512",512)] {
            if units<=maximum {candidates.push((scalar,(units,false)));candidates.push((vector,(units,true)));}
        }
        let output=tuning::execute_variants_prepared(vec![gate.clone(),up.clone()],"swiglu_inference_v1",
            "fp32_silu_storage_round_fp32_multiply_storage_round".into(),candidates,|values,config| {
                swiglu_activation_inner(values[0].clone(),values[1].clone(),config)
                    .map(|output|vec![output]).map_err(|error|error.to_string())
            }).map_err(|_|MoeError("native SwiGLU autotune failed without replay"))?;
        if let Some(mut output)=output {return Ok(output.remove(0));}
    }
    swiglu_activation_reference(gate,up)
}

pub fn swiglu_activation_reference<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {
    swiglu_activation_inner(gate,up,(0,false))
}

fn swiglu_activation_inner<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>,config:(u32,bool)) -> Result<RudaTensor<R>,MoeError> {
    let size=validate(&gate,&up)?;let output=empty(&gate,gate.meta.shape().clone(),gate.dtype);
    if size!=0 {
        let gate=into_contiguous(gate);let up=into_contiguous(up);
        let mut vector_size=if config.1 && gate.meta.num_dims()!=0 {max_vector_size(&gate).min(max_vector_size(&up)).min(max_vector_size(&output))} else {1};
        let alignment=vector_size as u64*gate.dtype.size() as u64;
        if [&gate,&up,&output].iter().any(|tensor|tensor.handle.offset_start.unwrap_or(0)%alignment!=0) {vector_size=1;}
        let units=size/vector_size as usize;
        let dim=if config.0==0 {RudaDim::new(gate.client.properties(),units)} else {RudaDim::new_1d(config.0)};
        let count=calculate_ruda_count_elemwise(&gate.client,units,dim);
        if config.1 {
            unsafe {
                kernels::experts::swiglu_vector::launch_unchecked::<R>(&gate.client,count,dim,
                    address_type!(gate,up,output),vector_size,gate.clone().into_linear_view_like(&output),
                    up.into_linear_view_like(&output),output.clone().into_linear_view(),gate.dtype.into());
            }
        } else {
            kernels::experts::swiglu_out::launch::<R>(&gate.client,count,dim,
                gate.clone().into_array_arg(),up.into_array_arg(),output.clone().into_array_arg(),gate.dtype.into());
        }
    }
    Ok(output)
}
/// Original first-order storage-rounded input VJPs, with only actual requested outputs.
pub fn swiglu_activation_backward<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>,gradient:RudaTensor<R>,selection:SwiGluActivationSelection)
    -> Result<SwiGluActivationBackward<R>,MoeError> {
    let size=validate(&gate,&up)?;validate(&gate,&gradient)?;
    let dg=selection.gate.then(||empty(&gate,gate.meta.shape().clone(),gate.dtype));
    let du=selection.up.then(||empty(&up,up.meta.shape().clone(),up.dtype));
    if size!=0 {let gate=into_contiguous(gate);let up=into_contiguous(up);let gradient=into_contiguous(gradient);let dim=RudaDim::new(gate.client.properties(),size);
        match (&dg,&du) {
            (Some(dg),Some(du))=>kernels::experts::swiglu_backward::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gradient.into_array_arg(),dg.clone().into_array_arg(),du.clone().into_array_arg(),gate.dtype.into()),
            (Some(dg),None)=>kernels::experts::swiglu_backward_gate::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gradient.into_array_arg(),dg.clone().into_array_arg(),gate.dtype.into()),
            (None,Some(du))=>kernels::experts::swiglu_backward_up::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),gradient.into_array_arg(),du.clone().into_array_arg(),gate.dtype.into()),
            (None,None)=>{},
        }
    }
    Ok(SwiGluActivationBackward {gate:dg,up:du})
}
