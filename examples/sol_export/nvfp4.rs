use std::path::PathBuf;
use ruda_core::ir::{FloatKind, IntKind, Type, UIntKind};
use rudnn::standalone::sol::nvfp4;
use super::Export;

pub fn export(d: &PathBuf, ix: &mut String) {
    let bf = Type::from(FloatKind::BF16);
    let fp = Type::from(FloatKind::F32);
    let byte = Type::from(UIntKind::U8);
    let mut e = Export::new("nv4_router6", 256, 1);
    let l = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false);
    let ids = e.buffer(Type::from(IntKind::I64), 1, true); let w = e.buffer(fp, 1, true);
    let t = e.scalar("I"); let n = e.scalar("I"); let scale = e.scalar("f");
    nvfp4::router6::expand(&mut e.kernel.body, l.into(), b.into(), ids.into(), w.into(), t.into(), n.into(), scale.into());
    e.save(d, ix);
    for (name, dtype) in [("nv4_amax_b", bf), ("nv4_amax_f", fp)] {
        let mut e = Export::new(name, 256, 1);
        let x = e.buffer(dtype, 8, false); let o = e.buffer(fp, 1, true);
        if dtype == bf { nvfp4::amax::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), dtype.storage_type()); }
        else { nvfp4::amax::expand::<f32>(&mut e.kernel.body, x.into(), o.into(), dtype.storage_type()); }
        e.save(d, ix);
    }
    for (name, dtype, rounded) in [("nv4_encode_b0", bf, false), ("nv4_encode_b1", bf, true), ("nv4_encode_f0", fp, false)] {
        let mut e = Export::new(name, 128, 1);
        let x = e.buffer(dtype, 8, false); let p = e.buffer(fp, 1, false);
        let o = e.buffer(bf, 8, true); let g = e.buffer(fp, 1, true);
        if dtype == bf { nvfp4::quantize::expand::<half::bf16>(&mut e.kernel.body, x.into(), p.into(), o.into(), g.into(), dtype.storage_type(), rounded); }
        else { nvfp4::quantize::expand::<f32>(&mut e.kernel.body, x.into(), p.into(), o.into(), g.into(), dtype.storage_type(), rounded); }
        e.save(d, ix);
    }
    let mut e = Export::new("nv4_decode", 256, 1);
    let x = e.buffer(byte, 1, false); let s = e.buffer(byte, 1, false); let o = e.buffer(bf, 8, true);
    nvfp4::dequantize::expand(&mut e.kernel.body, x.into(), s.into(), o.into());
    e.save(d, ix);
    let mut e = Export::new("nv4_vision_positions", 256, 1);
    let mask = e.buffer(byte, 1, false); let o = e.buffer(Type::from(IntKind::I32), 1, true);
    let seq = e.scalar("I"); let v = e.scalar("I");
    nvfp4::vision_positions::expand(&mut e.kernel.body, mask.into(), o.into(), seq.into(), v.into());
    e.save(d, ix);
    let mut e = Export::new("nv4_vision_merge", 256, 1);
    let ids = e.buffer(Type::from(IntKind::I64), 1, false); let p = e.buffer(Type::from(IntKind::I32), 1, false);
    let w = e.buffer(bf, 8, false); let v = e.buffer(bf, 8, false); let o = e.buffer(bf, 8, true);
    let seq = e.scalar("I"); let vr = e.scalar("I"); let width = e.scalar("I");
    nvfp4::vision_merge::expand(&mut e.kernel.body, ids.into(), p.into(), w.into(), v.into(), o.into(), seq.into(), vr.into(), width.into());
    e.save(d, ix);
    for (name, masked, causal) in [("bf16_scores_00", false, false), ("bf16_scores_01", false, true),
        ("bf16_scores_10", true, false), ("bf16_scores_11", true, true)] {
        let mut e = Export::new(name, 256, 1);
        let x = e.buffer(bf, 1, false); let mask = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true);
        let c = e.scalar("I"); let n = e.scalar("I"); let ld = e.scalar("I"); let q = e.scalar("I"); let h = e.scalar("I"); let scale = e.scalar("f");
        nvfp4::attention_scores::expand(&mut e.kernel.body, x.into(), mask.into(), o.into(), c.into(), n.into(), ld.into(), q.into(), h.into(), scale.into(), masked, causal);
        e.save(d, ix);
    }
}
