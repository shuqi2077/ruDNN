//! Exports for L1 batch C kernels (001, 005, 006, 008, 009).
use std::path::PathBuf;
use ruda_core::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use rudnn::standalone::sol::l1c;
use super::Export;

pub fn atomic_f32() -> Type { Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))) }

pub fn export(d: &PathBuf, ix: &mut String) {
    let bf = Type::from(FloatKind::BF16);
    let fp = Type::from(FloatKind::F32);
    let long = Type::from(IntKind::I64);
    let byte = Type::from(UIntKind::U8);

    let mut e = Export::new("softmax_bwd_bf16", 256, 1);
    let dp = e.buffer(fp, 1, false); let p = e.buffer(bf, 1, false); let k = e.buffer(byte, 1, false); let o = e.buffer(bf, 1, true);
    let cols = e.scalar("I"); let ld = e.scalar("I"); let drop = e.scalar("I"); let inv = e.scalar("f");
    l1c::softmax_bwd_rows::expand::<half::bf16>(&mut e.kernel.body, dp.into(), p.into(), k.into(), o.into(), cols.into(), ld.into(),
        drop.into(), inv.into(), bf.storage_type());
    e.save(d, ix);

    for (name, t) in [("pad_rows_bf16", bf), ("pad_rows_f32", fp)] {
        let mut e = Export::new(name, 256, 1);
        let x = e.buffer(t, 1, false); let o = e.buffer(t, 1, true);
        let count = e.scalar("I"); let cols = e.scalar("I"); let ld = e.scalar("I");
        if t == bf {
            l1c::pad_rows::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), count.into(), cols.into(), ld.into(), t.storage_type());
        } else {
            l1c::pad_rows::expand::<f32>(&mut e.kernel.body, x.into(), o.into(), count.into(), cols.into(), ld.into(), t.storage_type());
        }
        e.save(d, ix);
    }

    let mut e = Export::new("l1_005_conv_gate", 256, 1);
    let x = e.buffer(bf, 1, false); let w = e.buffer(bf, 1, false); let b = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true);
    let count = e.scalar("I"); let seq = e.scalar("I"); let hidden = e.scalar("I");
    l1c::conv_gate::expand::<half::bf16>(&mut e.kernel.body, x.into(), w.into(), b.into(), o.into(), count.into(), seq.into(), hidden.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_006_hyena", 256, 1);
    let u = e.buffer(fp, 1, false); let w = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false);
    let g = e.buffer(fp, 1, true); let x0 = e.buffer(fp, 1, true); let x1 = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let len = e.scalar("I");
    l1c::hyena_short_conv::expand(&mut e.kernel.body, u.into(), w.into(), b.into(), g.into(), x0.into(), x1.into(), count.into(), len.into());
    e.save(d, ix);

    let mut e = Export::new("index_add_bf16", 256, 1);
    let s = e.buffer(bf, 1, false); let i = e.buffer(long, 1, false); let a = e.buffer(atomic_f32(), 1, true);
    let count = e.scalar("I"); let width = e.scalar("I");
    l1c::index_add_rows::expand::<half::bf16>(&mut e.kernel.body, s.into(), i.into(), a.into(), count.into(), width.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("scatter_fb", 256, 1);
    let s = e.buffer(fp, 8, false); let i = e.buffer(long, 1, false); let o = e.buffer(bf, 8, true);
    let count = e.scalar("I"); let vectors = e.scalar("I");
    l1c::scatter_rows::expand::<f32, half::bf16>(&mut e.kernel.body, s.into(), i.into(), o.into(), count.into(), vectors.into(),
        fp.storage_type(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_009_prep", 256, 1);
    let g = e.buffer(bf, 8, false); let i = e.buffer(long, 1, false); let x = e.buffer(bf, 8, false); let w = e.buffer(bf, 1, false);
    let r = e.buffer(bf, 1, true); let ge = e.buffer(bf, 8, true); let vectors = e.scalar("I");
    l1c::moe_scatter_prep::expand::<half::bf16>(&mut e.kernel.body, g.into(), i.into(), x.into(), w.into(), r.into(), ge.into(), vectors.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_029_conv_gate", 256, 1);
    let p = e.buffer(bf, 1, false); let m = e.buffer(bf, 1, false); let w = e.buffer(bf, 1, false); let b = e.buffer(bf, 1, false);
    let o = e.buffer(bf, 1, true); let count = e.scalar("I"); let seq = e.scalar("I"); let ch = e.scalar("I");
    rudnn::standalone::sol::l1d::mamba_conv_gate::expand::<half::bf16>(&mut e.kernel.body, p.into(), m.into(), w.into(), b.into(), o.into(),
        count.into(), seq.into(), ch.into(), bf.storage_type());
    e.save(d, ix);

    use rudnn::standalone::sol::l1d;
    let mut e = Export::new("l1_042_add", 256, 1);
    let x = e.buffer(fp, 1, false); let v = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let cols = e.scalar("I"); let s = e.scalar("f");
    l1d::add_row_vector::expand(&mut e.kernel.body, x.into(), v.into(), o.into(), count.into(), cols.into(), s.into());
    e.save(d, ix);
    let mut e = Export::new("grn_sumsq", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true); let sp = e.scalar("I"); let ch = e.scalar("I");
    l1d::grn_sumsq::expand(&mut e.kernel.body, x.into(), o.into(), sp.into(), ch.into());
    e.save(d, ix);
    let mut e = Export::new("grn_scale", 512, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true); let eps = e.scalar("f");
    l1d::grn_scale::expand(&mut e.kernel.body, x.into(), o.into(), eps.into());
    e.save(d, ix);
    let mut e = Export::new("grn_apply", 256, 1);
    let x = e.buffer(fp, 1, false); let n = e.buffer(fp, 1, false); let w = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false);
    let o = e.buffer(fp, 1, true); let count = e.scalar("I"); let sp = e.scalar("I"); let ch = e.scalar("I");
    l1d::grn_apply::expand(&mut e.kernel.body, x.into(), n.into(), w.into(), b.into(), o.into(), count.into(), sp.into(), ch.into());
    e.save(d, ix);

    let mut e = Export::new("l1_051_qknorm_bwd", 256, 1);
    let g = e.buffer(bf, 8, false); let n = e.buffer(bf, 8, false); let r = e.buffer(fp, 1, false); let w = e.buffer(bf, 8, false);
    let o = e.buffer(bf, 8, true); let p = e.buffer(fp, 1, true);
    let rows = e.scalar("I"); let heads = e.scalar("I"); let seq = e.scalar("I");
    l1d::qknorm_bwd::expand::<half::bf16>(&mut e.kernel.body, g.into(), n.into(), r.into(), w.into(), o.into(), p.into(),
        rows.into(), heads.into(), seq.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_052_collapse", 256, 1);
    let h0 = e.buffer(bf, 1, false); let p1 = e.buffer(fp, 1, false); let p2 = e.buffer(fp, 1, false); let o = e.buffer(bf, 1, true);
    let width = e.scalar("I"); let eps = e.scalar("f");
    l1d::altup_collapse::expand::<half::bf16>(&mut e.kernel.body, h0.into(), p1.into(), p2.into(), o.into(), width.into(), eps.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_053_sparse", 256, 1);
    let x = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true); let width = e.scalar("I"); let mult = e.scalar("f");
    l1d::gauss_sparse::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), width.into(), mult.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("shift_ids", 256, 1);
    let x = e.buffer(long, 1, false); let o = e.buffer(long, 1, true); let count = e.scalar("I"); let seq = e.scalar("I");
    l1d::shift_ids::expand(&mut e.kernel.body, x.into(), o.into(), count.into(), seq.into());
    e.save(d, ix);

    let mut e = Export::new("l1_070_prefix", 256, 1);
    let a = e.buffer(bf, 1, false); let p = e.buffer(fp, 1, true); let rows = e.scalar("I");
    l1d::ssd_prefix::expand::<half::bf16>(&mut e.kernel.body, a.into(), p.into(), rows.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_070_mask", 256, 1);
    let g = e.buffer(fp, 1, false); let p = e.buffer(fp, 1, false); let o = e.buffer(bf, 1, true);
    let count = e.scalar("I"); let chunks = e.scalar("I");
    l1d::ssd_mask::expand::<half::bf16>(&mut e.kernel.body, g.into(), p.into(), o.into(), count.into(), chunks.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_072_group_sum", 256, 1);
    let g = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let s: Vec<_> = (0..5).map(|_| e.scalar("I")).collect();
    l1d::group_sum_transpose::expand(&mut e.kernel.body, g.into(), o.into(), s[0].clone().into(), s[1].clone().into(),
        s[2].clone().into(), s[3].clone().into(), s[4].clone().into());
    e.save(d, ix);
    let mut e = Export::new("l1_076_glu", 256, 1);
    let g = e.buffer(fp, 1, false); let r = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let tokens = e.scalar("I"); let inter = e.scalar("I"); let experts = e.scalar("I");
    let alpha = e.scalar("f"); let limit = e.scalar("f");
    l1d::oss_glu::expand(&mut e.kernel.body, g.into(), r.into(), o.into(), count.into(), tokens.into(), inter.into(), experts.into(),
        alpha.into(), limit.into());
    e.save(d, ix);

    let mut e = Export::new("l1_079_ew", 256, 1);
    let g = e.buffer(bf, 1, false); let ga = e.buffer(bf, 1, false); let s = e.buffer(bf, 1, false); let gt = e.buffer(bf, 1, false);
    let o1 = e.buffer(bf, 1, true); let o2 = e.buffer(bf, 1, true); let count = e.scalar("I"); let silu = e.scalar("I");
    l1d::gate_mul_bwd::expand::<half::bf16>(&mut e.kernel.body, g.into(), ga.into(), s.into(), gt.into(), o1.into(), o2.into(),
        count.into(), silu.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_087_rms_bwd", 256, 1);
    let g = e.buffer(bf, 1, false); let h = e.buffer(fp, 1, false); let r = e.buffer(fp, 1, false); let w = e.buffer(bf, 1, false);
    let ids = e.buffer(long, 1, false); let t = e.buffer(atomic_f32(), 1, true); let p = e.buffer(fp, 1, true); let rows = e.scalar("I");
    l1d::embed_rms_bwd::expand::<half::bf16>(&mut e.kernel.body, g.into(), h.into(), r.into(), w.into(), ids.into(), t.into(), p.into(),
        rows.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_091_qkrope_bwd", 128, 1);
    let ins: Vec<_> = (0..11).map(|_| e.buffer(bf, 1, false)).collect();
    let o1 = e.buffer(bf, 1, true); let o2 = e.buffer(bf, 1, true); let o3 = e.buffer(bf, 1, true); let p = e.buffer(fp, 1, true);
    let rows = e.scalar("I"); let seq = e.scalar("I");
    let v = |i: usize| ins[i].clone().into();
    l1d::qkrope_bwd::expand::<half::bf16>(&mut e.kernel.body, v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10),
        o1.into(), o2.into(), o3.into(), p.into(), rows.into(), seq.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l1_093_router_bwd", 192, 1);
    let g = e.buffer(bf, 1, false); let tw = e.buffer(bf, 1, false); let i = e.buffer(long, 1, false); let den = e.buffer(bf, 1, false);
    let s = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true); let experts = e.scalar("I"); let factor = e.scalar("f");
    l1d::router_bwd::expand::<half::bf16>(&mut e.kernel.body, g.into(), tw.into(), i.into(), den.into(), s.into(), o.into(),
        experts.into(), factor.into(), bf.storage_type());
    e.save(d, ix);

    use rudnn::standalone::sol::l2a;
    let mut e = Export::new("l2_031_temb", 256, 1);
    let t = e.buffer(fp, 1, false); let f = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let half = e.scalar("I"); let scale = e.scalar("f");
    l2a::timestep_embed::expand(&mut e.kernel.body, t.into(), f.into(), o.into(), count.into(), half.into(), scale.into());
    e.save(d, ix);
    let mut e = Export::new("l2_068_gelu_bwd", 256, 1);
    let g = e.buffer(fp, 1, false); let x = e.buffer(fp, 1, false); let t = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let count = e.scalar("I");
    l2a::gelu_bwd_given::expand(&mut e.kernel.body, g.into(), x.into(), t.into(), o.into(), count.into());
    e.save(d, ix);
    let mut e = Export::new("l2_035_dwconv", 256, 1);
    let x = e.buffer(fp, 1, false); let w = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let ch = e.scalar("I"); let hh = e.scalar("I"); let ww = e.scalar("I");
    l2a::dwconv7_nhwc::expand(&mut e.kernel.body, x.into(), w.into(), b.into(), o.into(), count.into(), ch.into(), hh.into(), ww.into());
    e.save(d, ix);
    let mut e = Export::new("grn_scale_n", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true); let ch = e.scalar("I"); let eps = e.scalar("f");
    l2a::grn_scale_n::expand(&mut e.kernel.body, x.into(), o.into(), ch.into(), eps.into());
    e.save(d, ix);

    let mut e = Export::new("l1_034_rope_freqs", 256, 1);
    let ids = e.buffer(fp, 1, false); let c = e.buffer(fp, 1, true); let s = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let theta = e.scalar("f");
    rudnn::standalone::sol::l1d::flux_rope_freqs::expand(&mut e.kernel.body, ids.into(), c.into(), s.into(), count.into(), theta.into());
    e.save(d, ix);

    // MoE routing
    use rudnn::standalone::sol::moe;
    let word = Type::from(UIntKind::U32);
    let atomic_u32 = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    let mut e = Export::new("moe_count", 256, 1);
    let ids = e.buffer(long, 1, false); let c = e.buffer(atomic_u32, 1, true); let count = e.scalar("I");
    moe::moe_count::expand(&mut e.kernel.body, ids.into(), c.into(), count.into());
    e.save(d, ix);
    let mut e = Export::new("moe_offsets", 32, 1);
    let c = e.buffer(word, 1, false); let g = e.buffer(word, 1, true); let cur = e.buffer(word, 1, true); let experts = e.scalar("I");
    moe::moe_offsets::expand(&mut e.kernel.body, c.into(), g.into(), cur.into(), experts.into());
    e.save(d, ix);
    let mut e = Export::new("moe_place", 256, 1);
    let ids = e.buffer(long, 1, false); let g = e.buffer(word, 1, false); let cur = e.buffer(atomic_u32, 1, true);
    let s = e.buffer(word, 1, true); let inv = e.buffer(word, 1, true); let count = e.scalar("I"); let k = e.scalar("I");
    moe::moe_place::expand(&mut e.kernel.body, ids.into(), g.into(), cur.into(), s.into(), inv.into(), count.into(), k.into());
    e.save(d, ix);
    let hp = Type::from(FloatKind::F16);
    for (name, t) in [("moe_gather_b", bf), ("moe_gather_h", hp), ("moe_gather_f", fp)] {
        let mut e = Export::new(name, 256, 1);
        let x = e.buffer(t, 8, false); let i = e.buffer(word, 1, false); let o = e.buffer(t, 8, true);
        let count = e.scalar("I"); let vectors = e.scalar("I");
        macro_rules! g { ($T:ty) => { moe::moe_gather::expand::<$T>(&mut e.kernel.body, x.into(), i.into(), o.into(), count.into(), vectors.into(), t.storage_type()) }; }
        if t == bf { g!(half::bf16) } else if t == hp { g!(half::f16) } else { g!(f32) }
        e.save(d, ix);
    }
    for (fc, wc, round, res) in [('b', 'b', true, false), ('b', 'b', true, true), ('b', 'b', false, false), ('b', 'f', false, false),
        ('b', 'f', true, false), ('b', 'f', false, true), ('f', 'f', false, false), ('h', 'h', true, false), ('h', 'f', false, false)] {
        let name = format!("moe_combine_{fc}{wc}_{}{}", u8::from(round), u8::from(res));
        let ft = match fc { 'b' => bf, 'h' => hp, _ => fp };
        let wt = match wc { 'b' => bf, 'h' => hp, _ => fp };
        let mut e = Export::new(&name, 256, 1);
        let y = e.buffer(ft, 8, false); let inv = e.buffer(word, 1, false); let w = e.buffer(wt, 1, false);
        let r = e.buffer(ft, 8, false); let o = e.buffer(ft, 8, true);
        let count = e.scalar("I"); let vectors = e.scalar("I"); let k = e.scalar("I");
        macro_rules! c { ($F:ty, $W:ty) => { moe::moe_combine::expand::<$F, $W>(&mut e.kernel.body, y.into(), inv.into(), w.into(), r.into(), o.into(),
            count.into(), vectors.into(), k.into(), ft.storage_type(), wt.storage_type(), round, res) }; }
        match (fc, wc) {
            ('b', 'b') => c!(half::bf16, half::bf16), ('b', _) => c!(half::bf16, f32),
            ('h', 'h') => c!(half::f16, half::f16), ('h', _) => c!(half::f16, f32), _ => c!(f32, f32),
        }
        e.save(d, ix);
    }

    let int = Type::from(IntKind::I32);
    let mut e = Export::new("sort_hist", 256, 1);
    let k = e.buffer(int, 1, false); let h = e.buffer(word, 1, true); let count = e.scalar("I");
    moe::sort_hist::expand(&mut e.kernel.body, k.into(), h.into(), count.into());
    e.save(d, ix);
    let mut e = Export::new("sort_scan", 256, 1);
    let h = e.buffer(word, 1, false); let b = e.buffer(word, 1, true); let o = e.buffer(word, 1, true); let chunks = e.scalar("I");
    moe::sort_scan::expand(&mut e.kernel.body, h.into(), b.into(), o.into(), chunks.into());
    e.save(d, ix);
    let mut e = Export::new("sort_scatter", 256, 1);
    let k = e.buffer(int, 1, false); let b = e.buffer(word, 1, false); let s = e.buffer(int, 1, true); let count = e.scalar("I");
    moe::sort_scatter::expand(&mut e.kernel.body, k.into(), b.into(), s.into(), count.into());
    e.save(d, ix);
    let mut e = Export::new("narrow_ids", 256, 1);
    let x = e.buffer(long, 1, false); let o = e.buffer(int, 1, true); let count = e.scalar("I");
    moe::narrow_ids::expand(&mut e.kernel.body, x.into(), o.into(), count.into());
    e.save(d, ix);
    let mut e = Export::new("cap_offsets", 32, 1);
    let o = e.buffer(word, 1, false); let g = e.buffer(word, 1, true); let experts = e.scalar("I"); let cap = e.scalar("I");
    moe::cap_offsets::expand(&mut e.kernel.body, o.into(), g.into(), experts.into(), cap.into());
    e.save(d, ix);
    let mut e = Export::new("cap_place", 256, 1);
    let s = e.buffer(int, 1, false); let o = e.buffer(word, 1, false); let g = e.buffer(word, 1, false);
    let t = e.buffer(word, 1, true); let inv = e.buffer(word, 1, true);
    let count = e.scalar("I"); let experts = e.scalar("I"); let cap = e.scalar("I"); let k = e.scalar("I");
    moe::cap_place::expand(&mut e.kernel.body, s.into(), o.into(), g.into(), t.into(), inv.into(), count.into(), experts.into(),
        cap.into(), k.into());
    e.save(d, ix);
    let mut e = Export::new("l2_005_ew", 256, 1);
    let ins: Vec<_> = (0..4).map(|_| e.buffer(bf, 1, false)).collect();
    let o1 = e.buffer(bf, 1, true); let o2 = e.buffer(bf, 1, true); let o3 = e.buffer(bf, 1, true); let count = e.scalar("I");
    rudnn::standalone::sol::l2a::swiglu_bwd_full::expand::<half::bf16>(&mut e.kernel.body, ins[0].clone().into(), ins[1].clone().into(),
        ins[2].clone().into(), ins[3].clone().into(), o1.into(), o2.into(), o3.into(), count.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("sigmoid_col_b", 256, 1);
    let l = e.buffer(bf, 1, false); let o = e.buffer(fp, 1, true); let count = e.scalar("I"); let ld = e.scalar("I");
    rudnn::standalone::sol::l2a::sigmoid_column::expand::<half::bf16>(&mut e.kernel.body, l.into(), o.into(), count.into(), ld.into(), bf.storage_type());
    e.save(d, ix);
    for (name, ft, wt, renorm, per_lane, biased) in [("softmax_topk_bb_1", bf, bf, true, 8usize, false),
        ("softmax_topk_bb_0", bf, bf, false, 8, false), ("softmax_topk16_bb_1", bf, bf, true, 16, false),
        ("softmax_topk_fb_b", fp, bf, true, 8, true), ("softmax_topk16_fb_1", fp, bf, true, 16, false)] {
        let mut e = Export::new(name, 256, 1);
        let l = e.buffer(ft, 1, false); let bias = e.buffer(fp, 1, false); let i = e.buffer(long, 1, true); let w = e.buffer(wt, 1, true);
        let tokens = e.scalar("I"); let experts = e.scalar("I"); let k = e.scalar("I"); let nm = e.scalar("f");
        if ft == bf {
            rudnn::standalone::sol::l2a::softmax_topk::expand::<half::bf16, half::bf16>(&mut e.kernel.body, l.into(), bias.into(), i.into(),
                w.into(), tokens.into(), experts.into(), k.into(), nm.into(), ft.storage_type(), wt.storage_type(), renorm, per_lane, biased);
        } else {
            rudnn::standalone::sol::l2a::softmax_topk::expand::<f32, half::bf16>(&mut e.kernel.body, l.into(), bias.into(), i.into(),
                w.into(), tokens.into(), experts.into(), k.into(), nm.into(), ft.storage_type(), wt.storage_type(), renorm, per_lane, biased);
        }
        e.save(d, ix);
    }
    for (name, wt, biased) in [("sigmoid_topk_b_1", bf, true), ("sigmoid_topk_b_0", bf, false), ("sigmoid_topk_f_1", fp, true),
        ("sigmoid_topk_f_0", fp, false)] {
        let mut e = Export::new(name, 256, 1);
        let l = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let i = e.buffer(long, 1, true); let w = e.buffer(wt, 1, true);
        let tokens = e.scalar("I"); let experts = e.scalar("I"); let k = e.scalar("I"); let factor = e.scalar("f");
        if wt == bf {
            rudnn::standalone::sol::l2a::sigmoid_topk::expand::<half::bf16>(&mut e.kernel.body, l.into(), b.into(), i.into(), w.into(),
                tokens.into(), experts.into(), k.into(), factor.into(), wt.storage_type(), biased);
        } else {
            rudnn::standalone::sol::l2a::sigmoid_topk::expand::<f32>(&mut e.kernel.body, l.into(), b.into(), i.into(), w.into(),
                tokens.into(), experts.into(), k.into(), factor.into(), wt.storage_type(), biased);
        }
        e.save(d, ix);
    }
    for (name, t) in [("l2_019_mrope", fp), ("mrope_b", bf)] {
        let mut e = Export::new(name, 256, 1);
        let x = e.buffer(t, 1, false); let o = e.buffer(t, 1, true);
        let count = e.scalar("I"); let seq = e.scalar("I"); let kv = e.scalar("I"); let batch = e.scalar("I");
        if t == fp {
            rudnn::standalone::sol::l2a::mrope_combine::expand::<f32>(&mut e.kernel.body, x.into(), o.into(), count.into(), seq.into(),
                kv.into(), batch.into(), t.storage_type());
        } else {
            rudnn::standalone::sol::l2a::mrope_combine::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), count.into(), seq.into(),
                kv.into(), batch.into(), t.storage_type());
        }
        e.save(d, ix);
    }
    let mut e = Export::new("rope_table_b", 256, 1);
    let p = e.buffer(long, 1, false); let inv = e.buffer(fp, 1, false); let c = e.buffer(bf, 1, true); let s = e.buffer(bf, 1, true);
    let count = e.scalar("I"); let dim = e.scalar("I"); let factor = e.scalar("f");
    rudnn::standalone::sol::l2a::rope_table_factor::expand::<half::bf16>(&mut e.kernel.body, p.into(), inv.into(), c.into(), s.into(),
        count.into(), dim.into(), factor.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("l2_049_route", 256, 1);
    let l = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let i = e.buffer(long, 1, true); let w = e.buffer(fp, 1, true);
    let tokens = e.scalar("I"); let factor = e.scalar("f");
    rudnn::standalone::sol::l2a::group_topk_route::expand(&mut e.kernel.body, l.into(), b.into(), i.into(), w.into(), tokens.into(), factor.into());
    e.save(d, ix);

    use rudnn::standalone::sol::quant;
    for given in [false, true] {
        let mut e = Export::new(&format!("fq_act_bb_{}", u8::from(given)), 256, 1);
        let x = e.buffer(bf, 1, false); let s = e.buffer(fp, 1, true); let o = e.buffer(bf, 1, true);
        let rows = e.scalar("I"); let cols = e.scalar("I");
        quant::fq_act::expand::<half::bf16, half::bf16>(&mut e.kernel.body, x.into(), s.into(), o.into(), rows.into(), cols.into(),
            bf.storage_type(), bf.storage_type(), given);
        e.save(d, ix);
        let mut e = Export::new(&format!("fq_act_fb_{}", u8::from(given)), 256, 1);
        let x = e.buffer(fp, 1, false); let s = e.buffer(fp, 1, true); let o = e.buffer(bf, 1, true);
        let rows = e.scalar("I"); let cols = e.scalar("I");
        quant::fq_act::expand::<f32, half::bf16>(&mut e.kernel.body, x.into(), s.into(), o.into(), rows.into(), cols.into(),
            fp.storage_type(), bf.storage_type(), given);
        e.save(d, ix);
        let mut e = Export::new(&format!("fq_weight_bb_{}", u8::from(given)), 256, 1);
        let w = e.buffer(bf, 8, false); let s = e.buffer(fp, 1, true); let o = e.buffer(bf, 8, true);
        let cols = e.scalar("I"); let sn = e.scalar("I"); let sk = e.scalar("I");
        quant::fq_weight::expand::<half::bf16, half::bf16>(&mut e.kernel.body, w.into(), s.into(), o.into(), cols.into(), sn.into(), sk.into(),
            bf.storage_type(), bf.storage_type(), given);
        e.save(d, ix);
    }
    let mut e = Export::new("q_014_yarn", 256, 1);
    let p = e.buffer(long, 1, false); let inv = e.buffer(fp, 1, false); let c = e.buffer(bf, 1, true); let s = e.buffer(bf, 1, true);
    let rows = e.scalar("I"); let ms = e.scalar("f");
    quant::yarn_cos_sin::expand::<half::bf16>(&mut e.kernel.body, p.into(), inv.into(), c.into(), s.into(), rows.into(), ms.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("q_023_ssm", 256, 1);
    let dt = e.buffer(bf, 1, false); let al = e.buffer(fp, 1, false); let bm = e.buffer(bf, 1, false);
    let o1 = e.buffer(bf, 1, true); let o2 = e.buffer(fp, 1, true); let o3 = e.buffer(fp, 1, true);
    let count = e.scalar("I"); let heads = e.scalar("I"); let groups = e.scalar("I"); let state = e.scalar("I");
    let lo = e.scalar("f"); let hi = e.scalar("f");
    quant::ssm_discretize::expand::<half::bf16>(&mut e.kernel.body, dt.into(), al.into(), bm.into(), o1.into(), o2.into(), o3.into(),
        count.into(), heads.into(), groups.into(), state.into(), lo.into(), hi.into(), bf.storage_type());
    e.save(d, ix);
    let mut e = Export::new("dq_fp8_b", 256, 1);
    let x = e.buffer(byte, 1, false); let s = e.buffer(fp, 1, false); let o = e.buffer(bf, 1, true);
    let count = e.scalar("I"); let cols = e.scalar("I"); let br = e.scalar("I"); let sr = e.scalar("I"); let sc = e.scalar("I");
    quant::dq_fp8::expand::<half::bf16>(&mut e.kernel.body, x.into(), s.into(), o.into(), count.into(), cols.into(), br.into(),
        sr.into(), sc.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("softmax_bwd_scaled_bf16", 256, 1);
    let g = e.buffer(bf, 1, false); let p = e.buffer(bf, 1, false); let o = e.buffer(bf, 1, true);
    let cols = e.scalar("I"); let ld = e.scalar("I"); let sc = e.scalar("f");
    l1c::softmax_bwd_scaled::expand::<half::bf16>(&mut e.kernel.body, g.into(), p.into(), o.into(), cols.into(), ld.into(), sc.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_017_mid", 256, 1);
    let g = e.buffer(fp, 1, false); let up = e.buffer(bf, 1, false); let gs = e.buffer(bf, 1, false); let gate = e.buffer(bf, 1, false);
    let sig = e.buffer(bf, 1, false); let gu = e.buffer(bf, 1, true); let gg = e.buffer(bf, 1, true); let count = e.scalar("I");
    l1c::swiglu_bwd_sig::expand::<half::bf16>(&mut e.kernel.body, g.into(), up.into(), gs.into(), gate.into(), sig.into(), gu.into(), gg.into(),
        count.into(), bf.storage_type());
    e.save(d, ix);

    let mut e = Export::new("l1_009_mid", 256, 1);
    let g = e.buffer(fp, 1, false); let gate = e.buffer(bf, 1, false); let up = e.buffer(bf, 1, false); let w1 = e.buffer(bf, 1, false);
    let gu = e.buffer(bf, 1, true); let gw = e.buffer(bf, 1, true); let count = e.scalar("I");
    l1c::moe_swiglu_bwd::expand::<half::bf16>(&mut e.kernel.body, g.into(), gate.into(), up.into(), w1.into(), gu.into(), gw.into(), count.into(), bf.storage_type());
    e.save(d, ix);
}
