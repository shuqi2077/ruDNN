use std::path::PathBuf;
use ruda_core::ir::{FloatKind, IntKind, UIntKind, Type};
use rudnn::standalone::sol::routing_ext;
use super::Export;
pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32); let bf = Type::from(FloatKind::BF16);
    let long = Type::from(IntKind::I64); let word = Type::from(UIntKind::U32);
    let mut e = Export::new("expert_pair_weights", 256, 1);
    let x = e.buffer(fp, 1, false); let ids = e.buffer(long, 1, false); let o = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let k = e.scalar("I");
    routing_ext::expert_pair_weights::expand(&mut e.kernel.body, x.into(), ids.into(), o.into(), n.into(), k.into()); e.save(d, ix);
    for (tag, ft) in [("f", fp), ("b", bf)] {
        let mut e = Export::new(&format!("scores_topk_{tag}"), 256, 1);
        let s = e.buffer(ft, 1, false); let i = e.buffer(long, 1, true); let w = e.buffer(ft, 1, true);
        let n = e.scalar("I"); let k = e.scalar("I"); let norm = e.scalar("I"); let eps = e.scalar("f"); let factor = e.scalar("f");
        if tag == "f" { routing_ext::topk::expand::<f32>(&mut e.kernel.body, s.into(), i.into(), w.into(), n.into(), k.into(), norm.into(), eps.into(), factor.into(), ft.storage_type()); }
        else { routing_ext::topk::expand::<half::bf16>(&mut e.kernel.body, s.into(), i.into(), w.into(), n.into(), k.into(), norm.into(), eps.into(), factor.into(), ft.storage_type()); }
        e.save(d, ix);
    }
    for (tag, ft, wt, rounded) in [("bb", bf, bf, true), ("bf", bf, fp, false), ("ff", fp, fp, false)] {
        let mut e = Export::new(&format!("ordered_combine_{tag}"), 256, 1);
        let y = e.buffer(ft, 1, false); let inv = e.buffer(word, 1, false); let ids = e.buffer(long, 1, false); let w = e.buffer(wt, 1, false); let o = e.buffer(ft, 1, true);
        let width = e.scalar("I"); let k = e.scalar("I");
        macro_rules! call { ($f:ty,$w:ty) => { routing_ext::ordered_combine::expand::<$f,$w>(&mut e.kernel.body, y.into(), inv.into(), ids.into(), w.into(), o.into(), width.into(), k.into(), ft.storage_type(), wt.storage_type(), rounded) }; }
        match tag { "bb" => call!(half::bf16,half::bf16), "bf" => call!(half::bf16,f32), _ => call!(f32,f32) }
        e.save(d, ix);
    }
    let mut e = Export::new("scatter_scores_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let ids = e.buffer(long, 1, false); let o = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let k = e.scalar("I");
    routing_ext::scatter_scores::expand(&mut e.kernel.body, x.into(), ids.into(), o.into(), n.into(), k.into()); e.save(d, ix);
    let mut e = Export::new("router_sparse_backward", 256, 1);
    let p = e.buffer(fp, 1, false); let ids = e.buffer(long, 1, false); let sums = e.buffer(fp, 1, false);
    let gw = e.buffer(bf, 1, false); let gm = e.buffer(bf, 1, false); let gl = e.buffer(bf, 1, false); let o = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let k = e.scalar("I");
    routing_ext::router_backward::expand(&mut e.kernel.body, p.into(), ids.into(), sums.into(), gw.into(), gm.into(), gl.into(), o.into(), n.into(), k.into()); e.save(d, ix);
    for reverse in [false, true] {
        let mut e = Export::new(if reverse {"pair_unpermute"} else {"pair_permute"}, 256, 1);
        let x = e.buffer(fp, 1, false); let inv = e.buffer(word, 1, false); let o = e.buffer(fp, 1, true);
        routing_ext::pair_permute::expand(&mut e.kernel.body, x.into(), inv.into(), o.into(), reverse); e.save(d, ix);
    }
}
