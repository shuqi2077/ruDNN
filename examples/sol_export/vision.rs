use std::path::PathBuf;
use ruda_core::ir::{FloatKind, IntKind, Type};
use rudnn::standalone::sol::vision;
use super::Export;

pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32);
    let hp = Type::from(FloatKind::F16);
    let bf = Type::from(FloatKind::BF16);
    let mut e = Export::new("position_rope_table_b", 256, 1);
    let p = e.buffer(Type::from(IntKind::I64), 1, false); let c = e.buffer(bf, 1, true); let s = e.buffer(bf, 1, true);
    let dim = e.scalar("I"); let theta = e.scalar("f");
    vision::position_rope_table::expand(&mut e.kernel.body, p.into(), c.into(), s.into(), dim.into(), theta.into()); e.save(d, ix);
    let mut e = Export::new("gather_scalar_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let ids = e.buffer(Type::from(IntKind::I64), 1, false); let o = e.buffer(fp, 1, true); let n = e.scalar("I");
    vision::gather_scalar::expand(&mut e.kernel.body, x.into(), ids.into(), o.into(), n.into()); e.save(d, ix);
    for backward in [false, true] {
        let mut e = Export::new(if backward {"relative_table_grad"} else {"relative_table"}, 256, 1);
        let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true); let n = e.scalar("I"); let dim = e.scalar("I");
        if backward { vision::relative_table_grad::expand(&mut e.kernel.body, x.into(), o.into(), n.into(), dim.into()); }
        else { vision::relative_table::expand(&mut e.kernel.body, x.into(), o.into(), n.into(), dim.into()); }
        e.save(d, ix);
    }
    let mut e = Export::new("mask_key_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let mask = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let h = e.scalar("I"); let q = e.scalar("I"); let k = e.scalar("I");
    vision::mask_key::expand(&mut e.kernel.body, x.into(), mask.into(), o.into(), h.into(), q.into(), k.into()); e.save(d, ix);
    for flag in [false, true] {
        let mut e = Export::new(&format!("rope_frequency_{}", flag as u8), 256, 1);
        let inv = e.buffer(fp, 1, false); let c = e.buffer(fp, 1, true); let s = e.buffer(fp, 1, true);
        let dim = e.scalar("I");
        vision::rope_from_frequency::expand(&mut e.kernel.body, inv.into(), c.into(), s.into(), dim.into(), flag);
        e.save(d, ix);
        let mut e = Export::new(&format!("multimodal_rope_{}", flag as u8), 256, 1);
        let x = e.buffer(bf, 1, false); let p = e.buffer(Type::from(IntKind::I64), 1, false); let o = e.buffer(bf, 1, true);
        let heads = e.scalar("I");
        vision::multimodal_rope::expand(&mut e.kernel.body, x.into(), p.into(), o.into(), heads.into(), flag);
        e.save(d, ix);
    }
    let mut e = Export::new("sequence_rope_b", 256, 1);
    let c = e.buffer(bf, 1, true); let s = e.buffer(bf, 1, true);
    let seq = e.scalar("I"); let dim = e.scalar("I"); let theta = e.scalar("f");
    vision::sequence_rope::expand::<half::bf16>(&mut e.kernel.body, c.into(), s.into(), seq.into(), dim.into(), theta.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("norm_arithmetic_h", 256, 1);
    let x = e.buffer(hp, 8, false); let w = e.buffer(hp, 8, false); let b = e.buffer(hp, 8, false); let o = e.buffer(hp, 8, true);
    let n = e.scalar("I"); let eps = e.scalar("f");
    vision::norm_arithmetic::expand::<half::f16>(&mut e.kernel.body, x.into(), w.into(), b.into(), o.into(), n.into(), eps.into(), hp.storage_type());
    e.save(d, ix);
    let mut e = Export::new("vision_grid_embedding", 256, 1);
    let t = e.buffer(fp, 1, false); let inv = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let c = e.buffer(fp, 1, true); let s = e.buffer(fp, 1, true);
    let f = e.scalar("I"); let h = e.scalar("I"); let w = e.scalar("I"); let channels = e.scalar("I"); let off = e.scalar("I");
    vision::grid_embedding::expand(&mut e.kernel.body, t.into(), inv.into(), o.into(), c.into(), s.into(),
        f.into(), h.into(), w.into(), channels.into(), off.into());
    e.save(d, ix);
    let mut e = Export::new("video_rope_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let t = e.buffer(fp, 1, false); let s = e.buffer(fp, 1, false);
    let q = e.buffer(fp, 1, true); let k = e.buffer(fp, 1, true); let v = e.buffer(fp, 1, true);
    let seq = e.scalar("I"); let patches = e.scalar("I"); let side = e.scalar("I");
    vision::video_rope::expand(&mut e.kernel.body, x.into(), t.into(), s.into(), q.into(), k.into(), v.into(), seq.into(), patches.into(), side.into());
    e.save(d, ix);
    for dim in [64usize, 80, 128] {
        let mut e = Export::new(&format!("varlen_f32_{dim}"), 128, 1);
        let q = e.buffer(fp, 1, false); let k = e.buffer(fp, 1, false); let v = e.buffer(fp, 1, false);
        let cu = e.buffer(Type::from(IntKind::I64), 1, false); let o = e.buffer(fp, 1, true);
        let heads = e.scalar("I"); let scale = e.scalar("f");
        vision::varlen_attention::expand(&mut e.kernel.body, q.into(), k.into(), v.into(), cu.into(), o.into(), heads.into(), scale.into(), dim);
        e.save(d, ix);
    }
}
