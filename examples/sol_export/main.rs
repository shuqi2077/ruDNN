//! Exports the RUDA kernels used by the SOL-ExecBench L1/L2/Quant collection
//! submissions as sm_100 PTX plus a launch manifest (same layout as
//! `flashinfer-export`, consumed by `package_sol.py`).
use std::{fs, path::PathBuf};
use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::{compiler::Compiler,
    ir::{FloatKind, IntKind, UIntKind, ManagedVariable, Scope, Type, Variable, VariableKind},
    kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility},
    launch::{ExecutionMode, RudaDim}};
use rudnn::standalone::sol;

mod generic;
mod l1c;
mod nvfp4;
mod vision;
mod spectral;
mod spatial;
mod scan_blocks;
mod routing_ext;
mod audio_relative;
mod multimodal_positions;

struct Export {
    kernel: KernelDefinition,
    widths: Vec<usize>,
    scalar_types: Vec<&'static str>,
    maps: Vec<(u32, usize, usize)>,
}

impl Export {
    fn new(name: &str, x: u32, y: u32) -> Self {
        let mut kernel = KernelDefinition {
            buffers: vec![], tensor_maps: vec![], scalars: vec![],
            ruda_dim: RudaDim::new_2d(x, y), body: Scope::root(false),
            options: KernelOptions { kernel_name: name.to_string(), ..Default::default() },
        };
        kernel.body.register_type::<usize>(UIntKind::U32.into());
        Self { kernel, widths: vec![], scalar_types: vec![], maps: vec![] }
    }
    fn tensor_map(&mut self, ty: Type, rows: usize, columns: usize) -> ManagedVariable {
        let id = self.widths.len() as u32;
        self.widths.push(1);
        self.maps.push((id, rows, columns));
        self.kernel.tensor_maps.push(KernelArg { id, visibility: Visibility::Read, ty, size: None, has_extended_meta: false });
        ManagedVariable::Plain(Variable::new(VariableKind::TensorMapInput(id), ty))
    }
    fn buffer(&mut self, ty: Type, width: usize, write: bool) -> ManagedVariable {
        let id = self.widths.len() as u32;
        let ty = ty.with_vector_size(width);
        self.widths.push(width);
        self.kernel.buffers.push(KernelArg { id, visibility: if write { Visibility::ReadWrite } else { Visibility::Read },
            ty, size: None, has_extended_meta: false });
        ManagedVariable::Plain(Variable::new(if write { VariableKind::GlobalOutputArray(id) } else { VariableKind::GlobalInputArray(id) }, ty))
    }
    fn scalar(&mut self, kind: &'static str) -> ManagedVariable {
        let ty = match kind { "I" => Type::from(UIntKind::U32), "i" => Type::from(IntKind::I32), "f" => Type::from(FloatKind::F32), _ => unreachable!() };
        let position = self.scalar_types.iter().position(|&t| t == kind).unwrap_or_else(|| {
            self.scalar_types.push(kind);
            self.kernel.scalars.push(ScalarKernelArg { ty: ty.storage_type(), count: 0 });
            self.scalar_types.len() - 1
        });
        let index = self.kernel.scalars[position].count;
        self.kernel.scalars[position].count += 1;
        ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(index as u32), ty))
    }
    fn save(self, directory: &PathBuf, index: &mut String) {
        let scalars = self.scalar_types.iter().zip(&self.kernel.scalars).map(|(t, a)| format!("{}:{}", t, a.count)).collect::<Vec<_>>().join(",");
        let widths = self.widths.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        let name = self.kernel.options.kernel_name.clone();
        let maps = self.maps.iter().map(|(id, r, c)| format!("{id}:{r}:{c}")).collect::<Vec<_>>().join(",");
        let options = PtxCompilationOptions { target: Some(PtxTarget { version: (8, 7), sm: 100 }) };
        let code = PtxCompiler.compile(self.kernel, &options, ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(code.dynamic_metadata_index.is_none());
        fs::write(directory.join(format!("{name}.ptx")), code.source).unwrap();
        index.push_str(&format!("{name}\t{}, {}, {}\t{}\t{widths}\t{scalars}\t{maps}\n", code.ruda_dim.x, code.ruda_dim.y, code.ruda_dim.z, code.shared_memory_bytes));
        println!("exported {name}");
    }
}

fn main() {
    let directory = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    fs::create_dir_all(&directory).unwrap();
    let mut index = String::new();
    let bf = Type::from(FloatKind::BF16);
    let fp = Type::from(FloatKind::F32);
    let long = Type::from(IntKind::I64);
    let word = Type::from(UIntKind::U32);
    let d = &directory;
    let ix = &mut index;

    // L1/011 RoPE cos/sin table
    let mut e = Export::new("l1_011_rope_table", 256, 1);
    let p = e.buffer(long, 1, false); let f = e.buffer(fp, 1, false); let o = e.buffer(bf, 8, true); let s = e.scalar("f");
    sol::rope_table::expand::<half::bf16>(&mut e.kernel.body, p.into(), f.into(), o.into(), s.into(), bf.storage_type());
    e.save(d, ix);

    // L1/012 cos/sin of duplicated frequencies
    let mut e = Export::new("l1_012_cos_sin", 256, 1);
    let f = e.buffer(fp, 4, false); let c = e.buffer(bf, 4, true); let o = e.buffer(bf, 4, true); let s = e.scalar("f");
    sol::cos_sin_halves::expand::<half::bf16>(&mut e.kernel.body, f.into(), c.into(), o.into(), s.into(), bf.storage_type());
    e.save(d, ix);

    // L1/013 residual RMSNorm backward (hidden 2560 = 320 threads x 8)
    let mut e = Export::new("l1_013_rms_bwd", 320, 1);
    let g = e.buffer(bf, 8, false); let n = e.buffer(fp, 4, false); let r = e.buffer(fp, 1, false); let w = e.buffer(fp, 4, false);
    let gx = e.buffer(bf, 8, true); let gr = e.buffer(bf, 8, true); let part = e.buffer(fp, 4, true);
    sol::rms_backward_rows::expand::<half::bf16>(&mut e.kernel.body, g.into(), n.into(), r.into(), w.into(), gx.into(), gr.into(), part.into(), 320usize, bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("col_sum", 256, 1);
    let part = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    sol::column_sum::expand(&mut e.kernel.body, part.into(), o.into());
    e.save(d, ix);

    // L1/014 RoPE position gradient
    let mut e = Export::new("l1_014_rope_pos_grad", 256, 1);
    let gc = e.buffer(bf, 1, false); let gs = e.buffer(bf, 1, false); let em = e.buffer(fp, 1, false); let inv = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let seq = e.scalar("I"); let s = e.scalar("f");
    sol::rope_position_grad::expand::<half::bf16>(&mut e.kernel.body, gc.into(), gs.into(), em.into(), inv.into(), o.into(), seq.into(), s.into(), bf.storage_type());
    e.save(d, ix);

    // L1/016 RoPE inverse frequencies
    let mut e = Export::new("l1_016_inv_freq", 64, 1);
    let o = e.buffer(fp, 1, true); let t = e.scalar("f");
    sol::rope_inverse_frequency::expand(&mut e.kernel.body, o.into(), t.into());
    e.save(d, ix);

    // L1/019 vision RoPE backward
    let mut e = Export::new("l1_019_rope_vision_bwd", 64, 1);
    let a = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let q = e.buffer(fp, 1, false); let k = e.buffer(fp, 1, false);
    let em = e.buffer(fp, 1, false); let gq = e.buffer(fp, 1, true); let gk = e.buffer(fp, 1, true); let ge = e.buffer(fp, 1, true);
    sol::rope_vision_backward::expand(&mut e.kernel.body, a.into(), b.into(), q.into(), k.into(), em.into(), gq.into(), gk.into(), ge.into());
    e.save(d, ix);

    // L1/024 2D vision RoPE inverse-frequency gradient (partials, then col_sum)
    let mut e = Export::new("l1_024_partial", 288, 1);
    let gc = e.buffer(fp, 1, false); let gs = e.buffer(fp, 1, false); let pos = e.buffer(long, 1, false); let em = e.buffer(fp, 1, false);
    let part = e.buffer(fp, 1, true);
    sol::rope_grid_frequency_partial::expand(&mut e.kernel.body, gc.into(), gs.into(), pos.into(), em.into(), part.into());
    e.save(d, ix);

    // L1/025 tanh GELU
    let mut e = Export::new("l1_025_gelu", 256, 1);
    let x = e.buffer(fp, 4, false); let o = e.buffer(fp, 4, true);
    sol::gelu_tanh_f32::expand(&mut e.kernel.body, x.into(), o.into());
    e.save(d, ix);

    // L1/028 hybrid attention masks
    let mut e = Export::new("l1_028_masks", 256, 1);
    let full = e.buffer(word, 4, true); let sliding = e.buffer(word, 4, true);
    let t = e.scalar("I"); let s = e.scalar("I"); let past = e.scalar("I");
    sol::hybrid_masks::expand(&mut e.kernel.body, full.into(), sliding.into(), t.into(), s.into(), past.into());
    e.save(d, ix);

    // RMSNorm rows: L1/033 (post residual, 4096) and L1/069 (pre residual, 8192)
    for (name, width, threads, pre, post) in [("rms_post_4096", 4096usize, 256u32, false, true), ("rms_pre_8192", 8192usize, 512u32, true, false)] {
        let mut e = Export::new(name, threads, 1);
        let x = e.buffer(bf, 8, false); let r = e.buffer(bf, 8, false); let w = e.buffer(bf, 8, false); let o = e.buffer(bf, 8, true);
        let eps = e.scalar("f");
        sol::rms_rows::expand::<half::bf16>(&mut e.kernel.body, x.into(), r.into(), w.into(), o.into(), eps.into(),
            width, threads as usize, pre, post, bf.storage_type());
        e.save(d, ix);
    }

    // L1/038 per-head RMSNorm of q and k
    let mut e = Export::new("l1_038_head_rms", 256, 1);
    let q = e.buffer(fp, 4, false); let k = e.buffer(fp, 4, false); let wq = e.buffer(fp, 4, false); let wk = e.buffer(fp, 4, false);
    let oq = e.buffer(fp, 4, true); let ok = e.buffer(fp, 4, true); let eps = e.scalar("f");
    sol::head_rms_qk::expand(&mut e.kernel.body, q.into(), k.into(), wq.into(), wk.into(), oq.into(), ok.into(), eps.into(), 48usize);
    e.save(d, ix);

    // L1/039 instance-norm backward
    let mut e = Export::new("l1_039_partial", 256, 1);
    let g = e.buffer(fp, 1, false); let x = e.buffer(fp, 1, false); let m = e.buffer(fp, 1, false); let part = e.buffer(fp, 1, true);
    let hw = e.scalar("I"); let chunk = e.scalar("I");
    sol::instance_norm_partial::expand(&mut e.kernel.body, g.into(), x.into(), m.into(), part.into(), hw.into(), chunk.into());
    e.save(d, ix);
    let mut e = Export::new("l1_039_finalize", 64, 1);
    let part = e.buffer(fp, 1, false); let w = e.buffer(fp, 1, false); let sd = e.buffer(fp, 1, false);
    let coef = e.buffer(fp, 1, true); let gw = e.buffer(fp, 1, true); let gb = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let chunks = e.scalar("I"); let hw = e.scalar("I");
    sol::instance_norm_finalize::expand(&mut e.kernel.body, part.into(), w.into(), sd.into(), coef.into(), gw.into(), gb.into(), n.into(), chunks.into(), hw.into());
    e.save(d, ix);
    let mut e = Export::new("l1_039_apply", 256, 1);
    let g = e.buffer(fp, 1, false); let x = e.buffer(fp, 1, false); let m = e.buffer(fp, 1, false); let coef = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let hw = e.scalar("I"); let chunk = e.scalar("I");
    sol::instance_norm_apply::expand(&mut e.kernel.body, g.into(), x.into(), m.into(), coef.into(), o.into(), hw.into(), chunk.into());
    e.save(d, ix);

    // L1/046 softcapped softmax
    let mut e = Export::new("l1_046_softcap_softmax", 256, 1);
    let x = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true); let n = e.scalar("I"); let cap = e.scalar("f");
    sol::softcap_softmax_rows::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), n.into(), cap.into(), bf.storage_type());
    e.save(d, ix);

    export_batch_b(d, ix);
    generic::export_all(d, ix);
    l1c::export(d, ix);
    nvfp4::export(d, ix);
    vision::export(d, ix);
    spectral::export(d, ix);
    spatial::export(d, ix);
    scan_blocks::export(d, ix);
    routing_ext::export(d, ix);
    audio_relative::export(d, ix);
    multimodal_positions::export(d, ix);
    fs::write(directory.join("manifest.tsv"), index).unwrap();
}

fn export_batch_b(d: &PathBuf, ix: &mut String) {
    let bf = Type::from(FloatKind::BF16);
    let fp = Type::from(FloatKind::F32);
    let double = Type::from(FloatKind::F64);
    let long = Type::from(IntKind::I64);
    let word = Type::from(UIntKind::U32);
    let byte = Type::from(UIntKind::U8);

    // L1/056 Mamba dt backward + bf16 head sums
    let mut e = Export::new("l1_056_dt_bwd", 320, 1);
    let g = e.buffer(bf, 1, false); let b = e.buffer(bf, 1, false); let a = e.buffer(bf, 1, false);
    let o = e.buffer(bf, 1, true); let part = e.buffer(fp, 1, true); let lo = e.scalar("f"); let hi = e.scalar("f");
    sol::dt_softplus_clamp_backward::expand::<half::bf16>(&mut e.kernel.body, g.into(), b.into(), a.into(), o.into(), part.into(), lo.into(), hi.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("col_sum_bf16", 256, 1);
    let part = e.buffer(fp, 1, false); let o = e.buffer(bf, 1, true);
    sol::column_sum_cast::expand::<half::bf16>(&mut e.kernel.body, part.into(), o.into(), bf.storage_type());
    e.save(d, ix);

    // L1/059 group-limited top-k mask
    let mut e = Export::new("l1_059_group_mask", 256, 1);
    let s = e.buffer(fp, 1, false); let m = e.buffer(fp, 1, true); let g = e.buffer(fp, 1, true);
    sol::group_topk_mask::expand(&mut e.kernel.body, s.into(), m.into(), g.into());
    e.save(d, ix);

    // L1/061 tanh-gated residual backward
    let mut e = Export::new("l1_061_gate_bwd", 256, 1);
    let g = e.buffer(bf, 8, false); let gate = e.buffer(bf, 1, false); let h = e.buffer(bf, 8, false); let m = e.buffer(bf, 1, false);
    let gr = e.buffer(bf, 8, true); let gh = e.buffer(bf, 8, true); let part = e.buffer(fp, 1, true);
    sol::tanh_gate_backward::expand::<half::bf16>(&mut e.kernel.body, g.into(), gate.into(), h.into(), m.into(), gr.into(), gh.into(), part.into());
    e.save(d, ix);
    let mut e = Export::new("l1_061_gate_finish", 256, 1);
    let part = e.buffer(fp, 1, false); let gate = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true);
    sol::tanh_gate_finish::expand::<half::bf16>(&mut e.kernel.body, part.into(), gate.into(), o.into(), bf.storage_type());
    e.save(d, ix);

    // L1/062 KV-cache RoPE backward
    let mut e = Export::new("position_flags", 256, 1);
    let p = e.buffer(long, 1, false); let f = e.buffer(word, 1, true);
    sol::position_flags::expand(&mut e.kernel.body, p.into(), f.into());
    e.save(d, ix);
    let mut e = Export::new("l1_062_cache_copy", 256, 1);
    let k = e.buffer(bf, 8, false); let v = e.buffer(bf, 8, false); let f = e.buffer(word, 1, false);
    let ko = e.buffer(bf, 8, true); let vo = e.buffer(bf, 8, true);
    sol::cache_copy_zero_rows::expand::<half::bf16>(&mut e.kernel.body, k.into(), v.into(), f.into(), ko.into(), vo.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_062_kv_rope_bwd", 64, 1);
    let gk = e.buffer(bf, 1, false); let gv = e.buffer(bf, 1, false); let ks = e.buffer(bf, 1, false);
    let c = e.buffer(bf, 1, false); let s = e.buffer(bf, 1, false); let p = e.buffer(long, 1, false);
    let ok = e.buffer(bf, 1, true); let ov = e.buffer(bf, 1, true); let oc = e.buffer(bf, 1, true); let os = e.buffer(bf, 1, true);
    let len = e.scalar("I");
    sol::kv_rope_backward::expand::<half::bf16>(&mut e.kernel.body, gk.into(), gv.into(), ks.into(), c.into(), s.into(), p.into(),
        ok.into(), ov.into(), oc.into(), os.into(), len.into(), bf.storage_type());
    e.save(d, ix);

    // L1/066 masked softmax + dropout backward
    let mut e = Export::new("l1_066_softmax_bwd", 256, 1);
    let g = e.buffer(fp, 1, false); let p = e.buffer(fp, 1, false); let m = e.buffer(byte, 1, false); let dm = e.buffer(byte, 1, false);
    let o = e.buffer(fp, 1, true); let n = e.scalar("I"); let h = e.scalar("I"); let use_dropout = e.scalar("I"); let inv = e.scalar("f");
    sol::softmax_dropout_backward::expand(&mut e.kernel.body, g.into(), p.into(), m.into(), dm.into(), o.into(), n.into(), h.into(), use_dropout.into(), inv.into());
    e.save(d, ix);

    // L1/068 RoPE GQA backward
    let mut e = Export::new("l1_068_rope_gqa_bwd", 64, 1);
    let gq = e.buffer(fp, 1, false); let gk = e.buffer(fp, 1, false); let q = e.buffer(fp, 1, false); let k = e.buffer(fp, 1, false);
    let c = e.buffer(fp, 1, false); let s = e.buffer(fp, 1, false);
    let oq = e.buffer(fp, 1, true); let ok = e.buffer(fp, 1, true); let oc = e.buffer(fp, 1, true); let os = e.buffer(fp, 1, true);
    let seq = e.scalar("I"); let heads = e.scalar("I"); let kv = e.scalar("I");
    sol::rope_gqa_backward::expand(&mut e.kernel.body, gq.into(), gk.into(), q.into(), k.into(), c.into(), s.into(),
        oq.into(), ok.into(), oc.into(), os.into(), seq.into(), heads.into(), kv.into());
    e.save(d, ix);

    // L1/071 KV-cache append with RoPE
    let mut e = Export::new("l1_071_kv_append", 256, 1);
    let ks = e.buffer(bf, 8, false); let vs = e.buffer(bf, 8, false); let c = e.buffer(bf, 8, false); let s = e.buffer(bf, 8, false);
    let kc = e.buffer(bf, 8, false); let vc = e.buffer(bf, 8, false); let ko = e.buffer(bf, 8, true); let vo = e.buffer(bf, 8, true);
    let cur = e.scalar("I"); let fresh = e.scalar("I"); let heads = e.scalar("I");
    sol::kv_cache_append_rope::expand::<half::bf16>(&mut e.kernel.body, ks.into(), vs.into(), c.into(), s.into(), kc.into(), vc.into(),
        ko.into(), vo.into(), cur.into(), fresh.into(), heads.into(), bf.storage_type());
    e.save(d, ix);

    // L1/078 GroupNorm
    let mut e = Export::new("l1_078_gn_partial", 256, 1);
    let x = e.buffer(fp, 1, false); let part = e.buffer(double, 1, true); let hw = e.scalar("I"); let chunk = e.scalar("I");
    sol::group_norm_partial::expand(&mut e.kernel.body, x.into(), part.into(), hw.into(), chunk.into());
    e.save(d, ix);
    let mut e = Export::new("l1_078_gn_apply", 256, 1);
    let x = e.buffer(fp, 1, false); let part = e.buffer(double, 1, false); let w = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let hw = e.scalar("I"); let chunk = e.scalar("I"); let groups = e.scalar("I"); let eps = e.scalar("f");
    sol::group_norm_apply::expand(&mut e.kernel.body, x.into(), part.into(), w.into(), b.into(), o.into(), hw.into(), chunk.into(), groups.into(), eps.into());
    e.save(d, ix);

    // L1/084 SiLU backward
    let mut e = Export::new("l1_084_silu_bwd", 256, 1);
    let g = e.buffer(fp, 1, false); let x = e.buffer(fp, 1, false); let s = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    sol::silu_backward_from_sigmoid::expand(&mut e.kernel.body, g.into(), x.into(), s.into(), o.into());
    e.save(d, ix);

    // L1/085 GEGLU
    let mut e = Export::new("l1_085_geglu", 256, 1);
    let x = e.buffer(fp, 4, false); let o = e.buffer(fp, 4, true);
    sol::geglu_tanh_f32::expand(&mut e.kernel.body, x.into(), o.into(), 5120usize);
    e.save(d, ix);

    // L1/088 RoPE application
    let mut e = Export::new("l1_088_rope_apply", 256, 1);
    let q = e.buffer(fp, 4, false); let k = e.buffer(fp, 4, false); let c = e.buffer(fp, 4, false); let s = e.buffer(fp, 4, false);
    let oq = e.buffer(fp, 4, true); let ok = e.buffer(fp, 4, true); let seq = e.scalar("I");
    sol::rope_apply_f32::expand(&mut e.kernel.body, q.into(), k.into(), c.into(), s.into(), oq.into(), ok.into(), seq.into());
    e.save(d, ix);

    // L1/090 RoPE theta backward
    let mut e = Export::new("l1_090_theta_bwd", 256, 1);
    let gc = e.buffer(bf, 4, false); let gs = e.buffer(bf, 4, false); let t = e.buffer(fp, 4, false); let o = e.buffer(fp, 4, true);
    sol::rope_theta_backward::expand::<half::bf16>(&mut e.kernel.body, gc.into(), gs.into(), t.into(), o.into(), bf.storage_type());
    e.save(d, ix);

    // L1/094 time-decay scan
    let mut e = Export::new("l1_094_time_decay", 128, 1);
    let td = e.buffer(fp, 1, false); let k = e.buffer(fp, 1, false); let tf = e.buffer(fp, 1, false); let v = e.buffer(fp, 1, false);
    let mi = e.buffer(fp, 1, false); let ni = e.buffer(fp, 1, false); let di = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let mo = e.buffer(fp, 1, true); let no = e.buffer(fp, 1, true); let dout = e.buffer(fp, 1, true);
    let seq = e.scalar("I");
    sol::time_decay_scan::expand(&mut e.kernel.body, td.into(), k.into(), tf.into(), v.into(), mi.into(), ni.into(), di.into(),
        o.into(), mo.into(), no.into(), dout.into(), seq.into());
    e.save(d, ix);
}
