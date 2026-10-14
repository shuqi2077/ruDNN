#![cfg_attr(not(feature = "kernel-ir"), no_std)]

#[cfg(feature = "cann")]
pub mod cann;

pub mod attention;

#[cfg(feature = "tensor-normalization")]
pub mod normalization;

#[cfg(feature = "tensor-gated-delta")]
pub mod gated_delta;

#[cfg(feature = "tensor-kda")]
pub mod kda;

#[cfg(any(feature = "tensor-kda", feature = "tensor-gated-delta"))]
pub mod sequence_graph;

#[cfg(feature = "tensor-moe")]
pub mod moe;

#[cfg(feature = "kernel-ir")]
pub mod convolution;

#[cfg(feature = "pooling")]
pub mod pooling;

#[cfg(any(feature = "pooling", feature = "interpolation"))]
mod indexing;

#[cfg(feature = "interpolation")]
pub mod interpolation;

#[cfg(feature = "grid-sample")]
pub mod grid_sample;

#[cfg(feature = "ctc")]
pub mod ctc;





#[cfg(feature = "tensor-paged-attention")]
pub mod paged_attention;

#[cfg(feature = "tensor-paged-attention")]
pub mod dense_attention;

#[cfg(feature = "tensor-paged-attention")]
pub mod varlen_attention;
#[cfg(feature = "tensor-paged-attention")]
pub mod ring_attention;
