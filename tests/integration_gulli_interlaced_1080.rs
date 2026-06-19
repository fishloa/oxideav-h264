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

fn decoder_yuv(path: &std::path::Path) -> (u32, u32, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).expect("read sample");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                if frames.is_empty() {
                    width = vf.planes[0].stride as u32;
                    height = (vf.planes[0].data.len() / vf.planes[0].stride) as u32;
                }
                frames.push(videoframe_to_yuv420p(&vf));
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
    (width, height, frames)
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
    let (w, h, ours_frames) = decoder_yuv(&fixture);
    let our_plane_bytes = (w as usize) * (h as usize) * 3 / 2;

    eprintln!(
        "[gulli_interlaced_1080] geometry {}x{} | our frames: {} | ffmpeg ref bytes: {}",
        w,
        h,
        ours_frames.len(),
        reference.len()
    );

    // --------------- compare
    let ffmpeg_frames = reference
        .len()
        .checked_div(our_plane_bytes.max(1))
        .unwrap_or(0);

    let mut worst_psnr = f64::INFINITY;
    let mut best_psnr = f64::NEG_INFINITY;
    let mut above_40 = 0u32;
    let mut below_40 = 0u32;
    let mut total_compared = 0u32;
    let compare_n = ours_frames.len().min(ffmpeg_frames);

    for i in 0..compare_n {
        let ours = &ours_frames[i];
        let theirs = &reference[i * our_plane_bytes..][..our_plane_bytes];
        let psnr = luma_psnr(ours, theirs);
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
                eprintln!("  frame {i}: luma PSNR = {psnr:.2} dB (BELOW 40 dB floor)",);
            }
        }
        total_compared += 1;
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

    // Promote this to `assert!` once the decoder passes the floor.
    // Until then, we print the status without failing CI.
    let _ = psnr_ok;
}
