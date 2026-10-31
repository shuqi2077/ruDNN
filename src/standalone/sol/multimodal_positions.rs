use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda(launch)]
pub fn compact(ids: &Array<i64>, mask: &Array<i64>, tokens: &mut Array<i64>, ranks: &mut Array<i64>, counts: &mut Array<i64>, seq: u32) {
    let batch = RUDA_POS_X as usize;
    if UNIT_POS == 0u32 {
        let base = batch * seq as usize;
        let mut valid = 0usize;
        let mut images = 0i64;
        let mut videos = 0i64;
        let mut previous = 0i64;
        for i in 0usize..seq as usize {
            ranks[base + i] = 1i64;
            if mask[base + i] == 1i64 {
                let token = ids[base + i];
                tokens[base + valid] = token;
                ranks[base + i] = valid as i64;
                valid += 1;
                if previous == 151652i64 {
                    if token == 151655i64 { images += 1i64; }
                    if token == 151656i64 { videos += 1i64; }
                }
                previous = token;
            }
        }
        counts[batch * 3] = valid as i64;
        counts[batch * 3 + 1] = images;
        counts[batch * 3 + 2] = videos;
    }
}

#[ruda(launch)]
pub fn positions(tokens: &Array<i64>, ranks: &Array<i64>, counts: &Array<i64>, images: &Array<i64>, videos: &Array<i64>,
    seconds: &Array<f32>, out: &mut Array<i64>, delta: &mut Array<i64>, seq: u32,
) {
    let batch = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = seq as usize;
    let stride = ranks.len();
    let base = batch * n;
    let valid = counts[batch * 3] as usize;
    if images.len() == 0 && videos.len() == 0 {
        let mut i = thread;
        while i < n {
            out[base + i] = ranks[base + i];
            out[stride + base + i] = ranks[base + i];
            out[2 * stride + base + i] = ranks[base + i];
            i += 256;
        }
        if thread == 0 {
            let mut maximum = valid as i64 - 1i64;
            if valid < n && maximum < 1i64 { maximum = 1i64; }
            delta[batch] = maximum + 1i64 - n as i64;
        }
    } else {
        let mut i = thread;
        while i < n {
            out[base + i] = 1i64; out[stride + base + i] = 1i64; out[2 * stride + base + i] = 1i64;
            i += 256;
        }
        sync_ruda();
        let mut image_index = 0usize;
        let mut video_index = 0usize;
        for b in 0usize..batch {
            image_index += counts[b * 3 + 1] as usize;
            video_index += counts[b * 3 + 2] as usize;
        }
        let mut remain_images = counts[batch * 3 + 1];
        let mut remain_videos = counts[batch * 3 + 2];
        let mut start = 0usize;
        let mut next_position = 0i64;
        while remain_images + remain_videos > 0i64 {
            let mut image_at = valid + 1;
            let mut video_at = valid + 1;
            for j in start..valid {
                if remain_images > 0i64 && image_at == valid + 1 && tokens[base + j] == 151655i64 { image_at = j; }
                if remain_videos > 0i64 && video_at == valid + 1 && tokens[base + j] == 151656i64 { video_at = j; }
            }
            let mut end = image_at;
            let mut frames = 0usize;
            let mut gh = 0usize;
            let mut gw = 0usize;
            let mut second = 0.0f32;
            if image_at < video_at {
                if image_index * 3 >= images.len() { break; }
                frames = images[image_index * 3] as usize;
                gh = images[image_index * 3 + 1] as usize / 2;
                gw = images[image_index * 3 + 2] as usize / 2;
                image_index += 1; remain_images -= 1i64;
            } else {
                if video_index * 3 >= videos.len() { break; }
                end = video_at;
                frames = videos[video_index * 3] as usize;
                gh = videos[video_index * 3 + 1] as usize / 2;
                gw = videos[video_index * 3 + 2] as usize / 2;
                second = seconds[video_index];
                video_index += 1; remain_videos -= 1i64;
            }
            let text = end - start;
            let mut j = thread;
            while j < text && start + j < valid {
                let value = next_position + j as i64;
                out[base + start + j] = value; out[stride + base + start + j] = value; out[2 * stride + base + start + j] = value;
                j += 256;
            }
            let vision_base = next_position + text as i64;
            let pixels = frames * gh * gw;
            let mut j = thread;
            while j < pixels && end + j < valid {
                let temporal = ((j / (gh * gw)) as f32 * second * 2.0f32) as i64;
                out[base + end + j] = vision_base + temporal;
                out[stride + base + end + j] = vision_base + (j / gw % gh) as i64;
                out[2 * stride + base + end + j] = vision_base + (j % gw) as i64;
                j += 256;
            }
            let last_time = ((frames - 1) as f32 * second * 2.0f32) as i64;
            next_position = vision_base + i64::max(last_time, i64::max(gh as i64 - 1i64, gw as i64 - 1i64)) + 1i64;
            start = end + pixels;
        }
        if start < valid {
            let mut j = start + thread;
            while j < valid {
                let value = next_position + (j - start) as i64;
                out[base + j] = value; out[stride + base + j] = value; out[2 * stride + base + j] = value;
                j += 256;
            }
            next_position += (valid - start) as i64;
        }
        if thread == 0 { delta[batch] = next_position - valid as i64; }
    }
}
