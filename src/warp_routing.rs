use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda]
pub(crate) fn probabilities<F: Float, B: Float>(
    logits: &Array<F>, bias: &Array<B>, probability: &mut Array<f32>, corrected: &mut Array<f32>,
    token: usize, #[comptime] groups: usize, #[comptime] has_bias: bool,
    #[comptime] stable_sigmoid: bool,
) {
    let lane = UNIT_POS_PLANE as usize;
    #[unroll]
    for g in 0usize..groups {
        let expert = g * 32 + lane;
        let x = f32::cast_from(logits[token * groups * 32 + expert]);
        let mut p = 0.0f32;
        if comptime!(stable_sigmoid) {
            if x >= 0.0 { p = 1.0f32 / (1.0f32 + (-x).exp()); }
            else { let e = x.exp(); p = e / (1.0f32 + e); }
        } else { p = 1.0f32 / (1.0f32 + (-x).exp()); }
        probability[g] = p;
        if comptime!(has_bias) { p += f32::cast_from(bias[expert]); }
        corrected[g] = p;
    }
}

#[ruda]
pub(crate) fn select(
    probability: &Array<f32>, corrected: &Array<f32>, indices: &mut Array<u32>,
    weights: &mut Array<f32>, #[comptime] groups: usize, #[comptime] selected_groups: usize,
    #[comptime] top_k: usize, #[comptime] group_top_two: bool,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let mut group_scores = Array::<f32>::new(groups);
    let mut kept = Array::<u32>::new(groups);
    let mut taken = Array::<u32>::new(groups);
    #[unroll]
    for g in 0usize..groups {
        let score = corrected[g];
        let mut first = plane_max(score);
        let mut second = 0.0f32;
        if comptime!(group_top_two) {
            let mut id = 32u32;
            if score == first { id = lane as u32; }
            let first_id = plane_min(id);
            let mut remainder = score;
            if lane == first_id as usize { remainder = f32::new(f32::NEG_INFINITY); }
            second = plane_max(remainder);
        }
        if plane_any(score != score) {
            first = plane_shuffle(score, 0u32);
            second = plane_shuffle(score, 1u32);
            if second > first { let previous = first; first = second; second = previous; }
            #[unroll]
            for j in 2usize..32 {
                let value = plane_shuffle(score, j as u32);
                if value > first { second = first; first = value; }
                else if value > second { second = value; }
            }
        }
        if comptime!(group_top_two) { first += second; }
        group_scores[g] = first;
        kept[g] = 0;
        taken[g] = 0;
    }
    for _slot in 0usize..selected_groups {
        let mut found = false;
        let mut best = 0.0f32;
        let mut id = 0usize;
        #[unroll]
        for g in 0usize..groups {
            if kept[g] == 0 && (!found || group_scores[g] > best) {
                found = true; best = group_scores[g]; id = g;
            }
        }
        kept[id] = 1;
    }
    let mut denominator = 0.0f32;
    for slot in 0usize..top_k {
        let mut local_best = f32::new(f32::NEG_INFINITY);
        let mut first_id = (groups * 32) as u32;
        #[unroll]
        for g in 0usize..groups {
            if kept[g] != 0 && taken[g] == 0 {
                first_id = u32::min(first_id, (g * 32 + lane) as u32);
                if corrected[g] > local_best { local_best = corrected[g]; }
            }
        }
        first_id = plane_min(first_id);
        let best = plane_max(local_best);
        let mut id = (groups * 32) as u32;
        let mut first_score = 0.0f32;
        #[unroll]
        for g in 0usize..groups {
            if first_id / 32 == g as u32 { first_score = corrected[g]; }
            if kept[g] != 0 && taken[g] == 0 && corrected[g] == best {
                id = u32::min(id, (g * 32 + lane) as u32);
            }
        }
        first_score = plane_shuffle(first_score, first_id % 32);
        id = plane_min(id);
        if first_score != first_score { id = first_id; }
        let mut p = 0.0f32;
        #[unroll]
        for g in 0usize..groups {
            if id / 32 == g as u32 { p = probability[g]; }
            if id == (g * 32 + lane) as u32 { taken[g] = 1; }
        }
        p = plane_shuffle(p, id % 32);
        indices[slot] = id;
        weights[slot] = p;
        denominator += p;
    }
    denominator
}
