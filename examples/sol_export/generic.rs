//! Name-driven export of the generic SOL kernels. Every quoted literal in the
//! solution sections / sol_lib.py that names a generic kernel is exported once:
//!   gemm_<I><O>_<bn>_<a_mn b_mn batched_a batched_b>_<bias act combine scale exact>
//!   flash_<I><O>_<dim>_<bn>_<stages>_<bhld causal window softcap mask lse>
//!   norm_<F><W><O>_<layer add weight bias one_plus cast>
//!   softmax_<F><O>_<mask><causal>      rope_<F><C>_<interleaved><exact>
//!   im2col_<F> gather_<F> colpart_<F> colfin_<O> rowsum_<F><O> cast_<F><O>
//!   permute_<F> bin_<F><G><O>_<op> tf32_split_a tf32_split_b
//! Type letters: h f16, b bf16, f f32, t tf32 (f32 storage), e e4m3 / g e5m2 (u8 storage).
use std::{collections::BTreeSet, fs, path::{Path, PathBuf}};
use ruda_core::ir::{FloatKind, IntKind, UIntKind, Type};
use rudnn::standalone::{sol::common, sol_attention, sol_gemm};
use super::Export;

const PREFIXES: [&str; 18] = ["gemm_", "flash_", "norm_", "softmax_", "rope_", "im2col_", "gather_", "colpart_",
    "colfin_", "rowsum_", "cast_", "permute_", "bin_", "tf32_split_", "hnr_", "scatterseq_", "un_", "adaln_"];

fn ty(c: char) -> Type {
    match c {
        'h' => Type::from(FloatKind::F16),
        'b' => Type::from(FloatKind::BF16),
        'f' | 't' => Type::from(FloatKind::F32),
        'e' | 'g' => Type::from(UIntKind::U8),
        _ => panic!("unknown type letter {c}"),
    }
}

macro_rules! float_type {
    ($c:expr, $T:ident, $body:block) => {
        match $c {
            'h' => { type $T = half::f16; $body }
            'b' => { type $T = half::bf16; $body }
            'f' | 't' => { type $T = f32; $body }
            other => panic!("unsupported float type {other}"),
        }
    };
}

macro_rules! input_type {
    ($c:expr, $T:ident, $body:block) => {
        match $c {
            'h' => { type $T = half::f16; $body }
            'b' => { type $T = half::bf16; $body }
            'f' | 't' => { type $T = f32; $body }
            'e' | 'g' => { type $T = u8; $body }
            other => panic!("unsupported input type {other}"),
        }
    };
}

fn literals(directory: &Path, names: &mut BTreeSet<String>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() { literals(&path, names); continue; }
        if path.extension().and_then(|e| e.to_str()) != Some("py") { continue; }
        let text = fs::read_to_string(&path).unwrap();
        for piece in text.split(['\'', '"']) {
            if PREFIXES.iter().any(|p| piece.starts_with(p))
                && piece.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                names.insert(piece.to_string());
            }
        }
    }
}

fn flag(s: &str, i: usize) -> bool { s.as_bytes()[i] == b'1' }
fn digit(s: &str, i: usize) -> u32 { (s.as_bytes()[i] - b'0') as u32 }

pub fn export_all(d: &PathBuf, ix: &mut String) {
    let mut names = BTreeSet::new();
    literals(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sol_export")), &mut names);
    for name in names {
        let parts: Vec<&str> = name.split('_').collect();
        let t: Vec<char> = parts.get(1).map(|p| p.chars().collect()).unwrap_or_default();
        // problem-specific kernels sharing a prefix (softmax_topk_*, rope_table_*) are exported explicitly
        if parts[0] != "tf32" && (t.is_empty() || !t.iter().all(|c| "hbfteg".contains(*c))) { continue; }
        match parts[0] {
            "gemm" => gemm(d, ix, &name, &parts, &t),
            "flash" => flash(d, ix, &name, &parts, &t),
            "norm" => norm(d, ix, &name, &parts, &t),
            "softmax" => softmax(d, ix, &name, &parts, &t),
            "rope" => rope(d, ix, &name, &parts, &t),
            "hnr" => head_norm_rope(d, ix, &name, &parts, &t),
            "tf32" => {
                let mut e = Export::new(&name, 256, 1);
                let fp = ty('f');
                let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
                let count = e.scalar("I"); let chunk = e.scalar("I"); let out_chunk = e.scalar("I"); let stride = e.scalar("I");
                common::tf32_split::expand(&mut e.kernel.body, x.into(), o.into(), count.into(), chunk.into(), out_chunk.into(),
                    stride.into(), name.ends_with('a'));
                e.save(d, ix);
            }
            _ => simple(d, ix, &name, &parts, &t),
        }
    }
}

fn gemm(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let (ic, oc) = (t[0], t[1]);
    let bn: usize = p[2].parse().unwrap();
    let (l, f) = (p[3], p[4]);
    let kind = match ic { 'h' => 0, 'b' => 1, 't' => 2, 'e' => 3, 'g' => 4, _ => panic!("{name}") };
    let bytes = match kind { 2 => 4, 3 | 4 => 1, _ => 2 };
    let stages = if bn >= 256 { 4 } else { 6 };
    let (it, ot, fp) = (ty(ic), ty(oc), ty('f'));
    let mut e = Export::new(name, 192, 1);
    let a = e.tensor_map(it, 128, 128 / bytes); let b = e.tensor_map(it, bn, 128 / bytes);
    let out = e.buffer(ot, 8, true); let bias = e.buffer(ot, 1, false); let residual = e.buffer(ot, 8, false);
    let sa = e.buffer(fp, 1, false); let sb = e.buffer(fp, 1, false);
    let groups = e.buffer(Type::from(UIntKind::U32), 1, false);
    let grouped = l.len() > 2 && l.as_bytes()[2] == b'g';
    // m n k batches k_groups ld o_div o_s1 o_s2 col_split col_stride bias_stride res_s1, then 7 coordinates for A and B
    let s: Vec<_> = (0..27).map(|_| e.scalar("I")).collect();
    let alpha = e.scalar("f"); let beta = e.scalar("f");
    let v = |i: usize| s[i].clone().into();
    input_type!(ic, I, { float_type!(oc, O, {
        sol_gemm::gemm_ws::expand::<I, O>(&mut e.kernel.body, a.into(), b.into(), out.into(), bias.into(), residual.into(),
            sa.into(), sb.into(), groups.into(), v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11), v(12),
            v(13), v(14), v(15), v(16), v(17), v(18), v(19), v(20), v(21), v(22), v(23), v(24), v(25), v(26),
            alpha.into(), beta.into(), it.storage_type(), ot.storage_type(), bn, stages, kind,
            flag(l, 0), flag(l, 1), digit(f, 0), digit(f, 1), digit(f, 2), digit(f, 3), digit(f, 4), grouped);
    }) });
    e.save(d, ix);
}

fn flash(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let (ic, oc) = (t[0], t[1]);
    let dim: usize = p[2].parse().unwrap();
    let bn: usize = p[3].parse().unwrap();
    let stages: usize = p[4].parse().unwrap();
    let f = p[5];
    let kind = match ic { 'h' => 0, 'b' => 1, 't' => 2, _ => panic!("{name}") };
    let bytes = if kind == 2 { 4 } else { 2 };
    let (it, ot, fp, byte) = (ty(ic), ty(oc), ty('f'), ty('e'));
    let mut e = Export::new(name, 128, 1);
    let bhld = flag(f, 0);
    let q = e.tensor_map(it, 128, 128 / bytes); let k = e.tensor_map(it, bn, 128 / bytes); let v = e.tensor_map(it, bn, 128 / bytes);
    let out = e.buffer(ot, 8, true); let lse = e.buffer(fp, 1, true); let mask = e.buffer(fp, 1, false); let flags = e.buffer(byte, 1, false);
    let s: Vec<_> = (0..8).map(|_| e.scalar("I")).collect();
    let scale = e.scalar("f"); let cap = e.scalar("f");
    float_type!(ic, I, { float_type!(oc, O, {
        sol_attention::flash_fwd::expand::<I, O>(&mut e.kernel.body, q.into(), k.into(), v.into(), out.into(), lse.into(),
            mask.into(), flags.into(), s[0].clone().into(), s[1].clone().into(), s[2].clone().into(), s[3].clone().into(),
            s[4].clone().into(), s[5].clone().into(), s[6].clone().into(), s[7].clone().into(), scale.into(), cap.into(),
            it.storage_type(), ot.storage_type(), dim, bn, stages, kind,
            bhld, flag(f, 1), flag(f, 2), flag(f, 3), digit(f, 4), flag(f, 5));
    }) });
    e.save(d, ix);
}

fn norm(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let f = p[2];
    let (ft, wt, ot) = (ty(t[0]), ty(t[1]), ty(t[2]));
    let mut e = Export::new(name, 256, 1);
    let x = e.buffer(ft, 8, false); let r = e.buffer(ft, 8, false); let w = e.buffer(wt, 8, false); let b = e.buffer(wt, 8, false);
    let o = e.buffer(ot, 8, true); let ro = e.buffer(ft, 8, true);
    let width = e.scalar("I"); let eps = e.scalar("f");
    float_type!(t[0], F, { float_type!(t[1], W, { float_type!(t[2], O, {
        common::norm_rows::expand::<F, W, O>(&mut e.kernel.body, x.into(), r.into(), w.into(), b.into(), o.into(), ro.into(),
            width.into(), eps.into(), ft.storage_type(), wt.storage_type(), ot.storage_type(),
            flag(f, 0), flag(f, 1), flag(f, 2), flag(f, 3), flag(f, 4), flag(f, 5));
    }) }) });
    e.save(d, ix);
}

fn softmax(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let f = p[2];
    let (ft, ot, fp, byte) = (ty(t[0]), ty(t[1]), ty('f'), ty('e'));
    let mut e = Export::new(name, 256, 1);
    let x = e.buffer(ft, 1, false); let m = e.buffer(fp, 1, false); let fl = e.buffer(byte, 1, false); let o = e.buffer(ot, 1, true);
    let columns = e.scalar("I"); let ld_in = e.scalar("I"); let ld_out = e.scalar("I"); let scale = e.scalar("f");
    let outer = e.scalar("I"); let inner = e.scalar("I"); let queries = e.scalar("I"); let offset = e.scalar("I");
    float_type!(t[0], F, { float_type!(t[1], O, {
        common::softmax_rows::expand::<F, O>(&mut e.kernel.body, x.into(), m.into(), fl.into(), o.into(), columns.into(),
            ld_in.into(), ld_out.into(), scale.into(),
            outer.into(), inner.into(), queries.into(), offset.into(), ft.storage_type(), ot.storage_type(), digit(f, 0), flag(f, 1));
    }) });
    e.save(d, ix);
}

fn rope(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let f = p[2];
    let (ft, ct) = (ty(t[0]), ty(t[1]));
    let mut e = Export::new(name, 256, 1);
    let x = e.buffer(ft, 1, false); let c = e.buffer(ct, 1, false); let s = e.buffer(ct, 1, false); let o = e.buffer(ft, 1, true);
    let count = e.scalar("I"); let dim = e.scalar("I"); let rot = e.scalar("I"); let div = e.scalar("I"); let modulo = e.scalar("I");
    float_type!(t[0], F, { float_type!(t[1], C, {
        common::rope_rows::expand::<F, C>(&mut e.kernel.body, x.into(), c.into(), s.into(), o.into(), count.into(), dim.into(),
            rot.into(), div.into(), modulo.into(), ft.storage_type(), ct.storage_type(), flag(f, 0), flag(f, 1));
    }) });
    e.save(d, ix);
}

fn head_norm_rope(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let dim: usize = p[2].parse().unwrap();
    let f = p[3];
    let (ft, wt, ct, fp) = (ty(t[0]), ty(t[1]), ty(t[2]), ty('f'));
    let long = Type::from(IntKind::I64);
    let mut e = Export::new(name, 256, 1);
    let x = e.buffer(ft, 1, false); let w = e.buffer(wt, 1, false); let bias = e.buffer(wt, 1, false);
    let c = e.buffer(ct, 1, false); let s = e.buffer(ct, 1, false);
    let pos = e.buffer(long, 1, false); let inv = e.buffer(fp, 1, false); let o = e.buffer(ft, 1, true);
    let rows = e.scalar("I"); let outer = e.scalar("I"); let inner = e.scalar("I"); let stride = e.scalar("I"); let eps = e.scalar("f");
    let half_table = f.len() > 4 && flag(f, 4);
    float_type!(t[0], F, { float_type!(t[1], W, { float_type!(t[2], C, {
        common::head_norm_rope::expand::<F, W, C>(&mut e.kernel.body, x.into(), w.into(), bias.into(), c.into(), s.into(), pos.into(), inv.into(),
            o.into(), rows.into(), outer.into(), inner.into(), stride.into(), eps.into(), ft.storage_type(), wt.storage_type(), ct.storage_type(),
            dim, digit(f, 0), flag(f, 1), flag(f, 2), flag(f, 3), half_table);
    }) }) });
    e.save(d, ix);
}

fn simple(d: &PathBuf, ix: &mut String, name: &str, p: &[&str], t: &[char]) {
    let fp = ty('f');
    let long = Type::from(IntKind::I64);
    let mut e = Export::new(name, 256, 1);
    match p[0] {
        "im2col" => {
            let ft = ty(t[0]);
            let x = e.buffer(ft, 1, false); let o = e.buffer(ft, 1, true);
            let s: Vec<_> = (0..15).map(|_| e.scalar("I")).collect();
            float_type!(t[0], F, {
                common::im2col::expand::<F>(&mut e.kernel.body, x.into(), o.into(), s[0].clone().into(), s[1].clone().into(),
                    s[2].clone().into(), s[3].clone().into(), s[4].clone().into(), s[5].clone().into(), s[6].clone().into(),
                    s[7].clone().into(), s[8].clone().into(), s[9].clone().into(), s[10].clone().into(), s[11].clone().into(),
                    s[12].clone().into(), s[13].clone().into(), s[14].clone().into(), ft.storage_type());
            });
        }
        "gather" => {
            let ft = ty(t[0]);
            let tb = e.buffer(ft, 8, false); let i = e.buffer(long, 1, false); let o = e.buffer(ft, 8, true);
            let count = e.scalar("I"); let vectors = e.scalar("I");
            float_type!(t[0], F, {
                common::gather_rows::expand::<F>(&mut e.kernel.body, tb.into(), i.into(), o.into(), count.into(), vectors.into(), ft.storage_type());
            });
        }
        "colpart" => {
            let ft = ty(t[0]);
            let x = e.buffer(ft, 1, false); let o = e.buffer(fp, 1, true);
            let rows = e.scalar("I"); let cols = e.scalar("I"); let chunk = e.scalar("I");
            float_type!(t[0], F, {
                common::col_partial::expand::<F>(&mut e.kernel.body, x.into(), o.into(), rows.into(), cols.into(), chunk.into(), ft.storage_type());
            });
        }
        "colfin" => {
            let ot = ty(t[0]);
            let x = e.buffer(fp, 1, false); let o = e.buffer(ot, 1, true);
            let cols = e.scalar("I"); let chunks = e.scalar("I");
            float_type!(t[0], O, {
                common::col_finish::expand::<O>(&mut e.kernel.body, x.into(), o.into(), cols.into(), chunks.into(), ot.storage_type());
            });
        }
        "rowsum" => {
            let (ft, ot) = (ty(t[0]), ty(t[1]));
            let x = e.buffer(ft, 1, false); let o = e.buffer(ot, 1, true);
            let cols = e.scalar("I"); let scale = e.scalar("f");
            float_type!(t[0], F, { float_type!(t[1], O, {
                common::row_sum::expand::<F, O>(&mut e.kernel.body, x.into(), o.into(), cols.into(), scale.into(), ft.storage_type(), ot.storage_type());
            }) });
        }
        "cast" => {
            let (ft, ot) = (ty(t[0]), ty(t[1]));
            let x = e.buffer(ft, 8, false); let o = e.buffer(ot, 8, true);
            let count = e.scalar("I");
            float_type!(t[0], F, { float_type!(t[1], O, {
                common::cast_vec::expand::<F, O>(&mut e.kernel.body, x.into(), o.into(), count.into(), ft.storage_type(), ot.storage_type());
            }) });
        }
        "permute" => {
            let ft = ty(t[0]);
            let x = e.buffer(ft, 1, false); let o = e.buffer(ft, 1, true);
            let s: Vec<_> = (0..8).map(|_| e.scalar("I")).collect();
            float_type!(t[0], F, {
                common::permute_copy::expand::<F>(&mut e.kernel.body, x.into(), o.into(), s[0].clone().into(), s[1].clone().into(),
                    s[2].clone().into(), s[3].clone().into(), s[4].clone().into(), s[5].clone().into(), s[6].clone().into(),
                    s[7].clone().into(), ft.storage_type());
            });
        }
        "bin" => {
            let (ft, gt, ot) = (ty(t[0]), ty(t[1]), ty(t[2]));
            let a = e.buffer(ft, 1, false); let b = e.buffer(gt, 1, false); let o = e.buffer(ot, 1, true);
            let s: Vec<_> = (0..12).map(|_| e.scalar("I")).collect();
            let alpha = e.scalar("f");
            float_type!(t[0], F, { float_type!(t[1], G, { float_type!(t[2], O, {
                common::binary4::expand::<F, G, O>(&mut e.kernel.body, a.into(), b.into(), o.into(),
                    s[0].clone().into(), s[1].clone().into(), s[2].clone().into(), s[3].clone().into(), s[4].clone().into(),
                    s[5].clone().into(), s[6].clone().into(), s[7].clone().into(), s[8].clone().into(), s[9].clone().into(),
                    s[10].clone().into(), s[11].clone().into(), alpha.into(),
                    ft.storage_type(), gt.storage_type(), ot.storage_type(), digit(p[2], 0));
            }) }) });
        }
        "un" => {
            let (ft, ot) = (ty(t[0]), ty(t[1]));
            let x = e.buffer(ft, 1, false); let o = e.buffer(ot, 1, true); let count = e.scalar("I");
            float_type!(t[0], F, { float_type!(t[1], O, {
                common::unary::expand::<F, O>(&mut e.kernel.body, x.into(), o.into(), count.into(), ft.storage_type(), ot.storage_type(), p[2].parse().unwrap());
            }) });
        }
        "adaln" => {
            let (ft, mt, ot) = (ty(t[0]), ty(t[1]), ty(t[2]));
            let x = e.buffer(ft, 1, false); let m = e.buffer(mt, 1, false); let o = e.buffer(ot, 1, true);
            let s: Vec<_> = (0..5).map(|_| e.scalar("I")).collect();
            let eps = e.scalar("f");
            float_type!(t[0], F, { float_type!(t[1], M, { float_type!(t[2], O, {
                common::adaln_rows::expand::<F, M, O>(&mut e.kernel.body, x.into(), m.into(), o.into(), s[0].clone().into(), s[1].clone().into(),
                    s[2].clone().into(), s[3].clone().into(), s[4].clone().into(), eps.into(), ft.storage_type(), mt.storage_type(), ot.storage_type());
            }) }) });
        }
        "scatterseq" => {
            let ft = ty(t[0]);
            let x = e.buffer(ft, 8, false); let i = e.buffer(long, 1, false); let o = e.buffer(ft, 8, true);
            let count = e.scalar("I"); let seq = e.scalar("I"); let length = e.scalar("I"); let vectors = e.scalar("I");
            float_type!(t[0], F, {
                common::scatter_seq::expand::<F>(&mut e.kernel.body, x.into(), i.into(), o.into(), count.into(), seq.into(),
                    length.into(), vectors.into(), ft.storage_type());
            });
        }
        other => panic!("unknown generic kernel {other} ({name})"),
    }
    e.save(d, ix);
}
