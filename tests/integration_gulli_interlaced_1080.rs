//! PAFF conformance test for `gulli_interlaced_1080.h264` — a
//! 1920×1080 interlaced broadcast H.264 clip (High@4.0, PAFF:
//! frame_mbs_only_flag=0, mb_adaptive_frame_field_flag=0), open-GOP
//! with recovery-point SEI, decoded vs ffmpeg as the pixel oracle.
//!
//! Gate: skips when ffmpeg is absent.
//!
//! This test is the escape-hatch for Issue #11. When the decoder
//! produces the correct field-picture reconstruction and field-DPB
//! output weaving, the `psnr_ok` assertion will toggle from soft to
//! hard.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Error, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

// --------------- ffmpeg plumbing ---------------

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ffmpeg_raw_yuv(path: &std::path::Path) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

// --------------- our decoder ---------------

fn decoder_yuv(path: &std::path::Path) -> Vec<(u32, u32, Vec<u8>)> {
    let bytes = std::fs::read(path).expect("read sample");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let mut frame_idx = 0usize;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let w = vf.planes[0].stride as u32;
                let h = (vf.planes[0].data.len() / vf.planes[0].stride) as u32;
                let yuv = videoframe_to_yuv420p(&vf);
                if frame_idx == 0 {
                    eprintln!(
                        "  frame0: {}x{} first16: {:?}",
                        w,
                        h,
                        &yuv[..16.min(yuv.len())]
                    );
                    let _ = std::fs::write("/tmp/ours.yuv", &yuv);
                }
                frames.push((w, h, yuv));
                frame_idx += 1;
            }
            Ok(other) => {
                eprintln!("  unexpected non-video frame: {:?}", other);
            }
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => {
                eprintln!("  receive_frame error: {e}");
                break;
            }
        }
    }
    frames
}

fn videoframe_to_yuv420p(vf: &oxideav_core::VideoFrame) -> Vec<u8> {
    assert_eq!(vf.planes.len(), 3, "yuv420p requires 3 planes");
    let w = vf.planes[0].stride;
    let h = vf.planes[0].data.len() / w;
    let cw = w / 2;
    let ch = h / 2;
    assert_eq!(vf.planes[1].stride, cw, "Cb stride must be half width");
    assert_eq!(vf.planes[2].stride, cw, "Cr stride must be half width");
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    pack_plane(&mut out, &vf.planes[0].data, vf.planes[0].stride, w, h);
    pack_plane(&mut out, &vf.planes[1].data, vf.planes[1].stride, cw, ch);
    pack_plane(&mut out, &vf.planes[2].data, vf.planes[2].stride, cw, ch);
    out
}

fn pack_plane(dst: &mut Vec<u8>, src: &[u8], stride: usize, cols: usize, rows: usize) {
    if stride == cols {
        dst.extend_from_slice(&src[..cols * rows]);
    } else {
        assert!(stride >= cols);
        for r in 0..rows {
            dst.extend_from_slice(&src[r * stride..r * stride + cols]);
        }
    }
}

// --------------- PSNR ---------------

fn luma_psnr(ours: &[u8], theirs: &[u8]) -> f64 {
    let n = ours.len().min(theirs.len());
    if n == 0 {
        return 0.0;
    }
    let mse = ours[..n]
        .iter()
        .zip(&theirs[..n])
        .map(|(&o, &t)| {
            let d = o as f64 - t as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if mse == 0.0 {
        100.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

// --------------- test ---------------

#[test]
fn conformance_gulli_interlaced_1080() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }

    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gulli_interlaced_1080.h264");
    if !fixture.exists() {
        eprintln!("skip: fixture not found at {}", fixture.display());
        return;
    }

    eprintln!("[gulli_interlaced_1080] decoding with ffmpeg …");
    let Some(reference) = ffmpeg_raw_yuv(&fixture) else {
        eprintln!("[gulli_interlaced_1080] skip: ffmpeg failed to decode");
        return;
    };

    eprintln!("[gulli_interlaced_1080] decoding with oxideav-h264 …");
    let ours_frames = decoder_yuv(&fixture);

    // ffmpeg produces 1920x1080 frames; our weaved frames are 1920x1088.
    // The ffmpeg planes have width=1920 (same as stride), height=1080.
    let ffmpeg_w = 1920usize;
    let ffmpeg_h = 1080usize;
    let ffmpeg_plane_bytes = ffmpeg_w * ffmpeg_h * 3 / 2;
    let ffmpeg_frames = reference
        .len()
        .checked_div(ffmpeg_plane_bytes.max(1))
        .unwrap_or(0);

    eprintln!(
        "[gulli_interlaced_1080] ffmpeg {}x{} | our frames: {} | ffmpeg ref bytes: {} (est. {} frames)",
        ffmpeg_w, ffmpeg_h, ours_frames.len(), reference.len(), ffmpeg_frames
    );

    // --------------- compare: match our frames against ffmpeg frames,
    // trimming our 1088-height frames to 1080 for comparison
    let mut worst_psnr = f64::INFINITY;
    let mut best_psnr = f64::NEG_INFINITY;
    let mut above_40 = 0u32;
    let mut below_40 = 0u32;
    let mut total_compared = 0u32;
    let max_compare = 20usize;
    let compare_n = ours_frames.len().min(ffmpeg_frames).min(max_compare);

    // Build a ffmpeg-frame index: advance through reference bytes per frame
    let mut ffmpeg_offset: usize = 0;
    let mut our_idx: usize = 0;
    while our_idx < compare_n {
        let (ow, oh, ref our_yuv) = &ours_frames[our_idx];
        let ow = *ow as usize;
        let oh = *oh as usize;
        // Trim our frame to ffmpeg dimensions
        let our_trimmed = if ow == ffmpeg_w && oh == ffmpeg_h {
            our_yuv.clone()
        } else if ow == ffmpeg_w {
            // Trim height (1088 → 1080)
            let h = oh.min(ffmpeg_h);
            let cw = ffmpeg_w / 2;
            let ch = h / 2;
            let mut out = Vec::with_capacity(ffmpeg_w * h + 2 * cw * ch);
            // Luma: first h rows
            for r in 0..h {
                out.extend_from_slice(&our_yuv[r * ow..r * ow + ffmpeg_w]);
            }
            // Cb/Cr: first ch rows each
            let cb_start = ow * oh;
            let cr_start = cb_start + cw * (oh / 2);
            for r in 0..ch {
                out.extend_from_slice(&our_yuv[cb_start + r * cw..cb_start + r * cw + cw]);
            }
            for r in 0..ch {
                out.extend_from_slice(&our_yuv[cr_start + r * cw..cr_start + r * cw + cw]);
            }
            out
        } else {
            // Different width — can't compare, skip
            our_idx += 1;
            ffmpeg_offset += ffmpeg_plane_bytes;
            continue;
        };
        let our_bytes = our_trimmed.len();
        let theirs = &reference[ffmpeg_offset..ffmpeg_offset + our_bytes.min(ffmpeg_plane_bytes)];

        let psnr = luma_psnr(&our_trimmed, theirs);
        if psnr < worst_psnr {
            worst_psnr = psnr;
        }
        if psnr > best_psnr {
            best_psnr = psnr;
        }
        if psnr >= 40.0 {
            above_40 += 1;
        } else {
            below_40 += 1;
            if below_40 <= 5 {
                eprintln!("  frame {our_idx}: luma PSNR = {psnr:.2} dB (BELOW 40 dB floor)",);
            }
        }
        total_compared += 1;
        our_idx += 1;
        ffmpeg_offset += ffmpeg_plane_bytes;
    }

    let psnr_ok = above_40 == total_compared && total_compared > 0;

    eprintln!(
        "[gulli_interlaced_1080] compared {total_compared} frames  | \
         above-40: {above_40}  below-40: {below_40}  |  \
         worst PSNR = {worst_psnr:.2} dB  best = {best_psnr:.2} dB  |  \
         PSNR status: {}",
        if psnr_ok { "PASS" } else { "FAIL" },
    );

    if !psnr_ok {
        eprintln!(
            "[gulli_interlaced_1080] NOTE: luma PSNR < 40 dB — \
             PAFF field-picture reconstruction is still under \
             development. The decoder currently produces {}/{} frames \
             above the 40 dB floor.  Issue #11.",
            above_40, total_compared,
        );
    }

    // Issue #11 exit gate — every compared frame must clear the 40 dB
    // luma PSNR floor vs ffmpeg. PAFF field-picture decode (field scans,
    // §8.2.4.2.5 field reference lists, frame-unit sliding-window
    // marking, open-GOP leading-picture suppression) brings the whole
    // sequence to ~53 dB.
    assert!(
        psnr_ok,
        "[gulli_interlaced_1080] luma PSNR floor not met: {above_40}/{total_compared} \
         frames above 40 dB (worst {worst_psnr:.2} dB). Issue #11.",
    );
}
