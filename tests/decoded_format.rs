//! The registry decoder reports the size and pixel layout of the frame it
//! last returned (oxideav-core `Decoder::output_video_dimensions` /
//! `output_pixel_format`): odd sizes and 10-bit included. A picture queued
//! behind a newer sequence header decodes and reports with its own header.
//!
//! §10.5.1 halves an odd subsampled luma size rounding down, where
//! `PixelFormat` rounds up: such a frame reports its size and no format.
//!
//! The streams come from the crate's own HQ intra encoder, one picture
//! each.

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_hq_intra_stream_u16, make_minimal_sequence,
    make_minimal_sequence_with_signal_range, EncoderParams,
};
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

/// A one-picture stream and the frame it decodes to.
struct Stream {
    bytes: Vec<u8>,
    size: (u32, u32),
    /// §10.5.1 chroma plane width × height.
    chroma: (u32, u32),
    bytes_per_sample: usize,
    /// `None` where the chroma planes match no `PixelFormat`.
    format: Option<PixelFormat>,
}

fn ramp(len: usize, seed: usize, max: usize) -> Vec<usize> {
    (0..len).map(|i| (i * 31 + seed * 7) % (max + 1)).collect()
}

fn stream_8bit(w: u32, h: u32, chroma: ChromaFormat, format: Option<PixelFormat>) -> Stream {
    let seq = make_minimal_sequence(w, h, chroma);
    let (cw, ch) = (seq.chroma_width, seq.chroma_height);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let plane = |len: u32, seed| -> Vec<u8> {
        ramp(len as usize, seed, 255)
            .into_iter()
            .map(|v| v as u8)
            .collect()
    };
    let bytes = encode_single_hq_intra_stream(
        &seq,
        &params,
        0,
        &plane(w * h, 1),
        &plane(cw * ch, 2),
        &plane(cw * ch, 3),
    );
    Stream {
        bytes,
        size: (w, h),
        chroma: (cw, ch),
        bytes_per_sample: 1,
        format,
    }
}

fn stream_10bit(w: u32, h: u32, chroma: ChromaFormat, format: Option<PixelFormat>) -> Stream {
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, SignalRange::PRESET_10BIT_FULL);
    let (cw, ch) = (seq.chroma_width, seq.chroma_height);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    // Deep coefficients overflow a scaler-1 HQ slice length byte.
    params.slice_size_scaler = 16;
    let plane = |len: u32, seed| -> Vec<u16> {
        ramp(len as usize, seed, 1023)
            .into_iter()
            .map(|v| v as u16)
            .collect()
    };
    let bytes = encode_single_hq_intra_stream_u16(
        &seq,
        &params,
        0,
        &plane(w * h, 1),
        &plane(cw * ch, 2),
        &plane(cw * ch, 3),
    );
    Stream {
        bytes,
        size: (w, h),
        chroma: (cw, ch),
        bytes_per_sample: 2,
        format,
    }
}

fn streams() -> [Stream; 3] {
    [
        stream_8bit(33, 17, ChromaFormat::Yuv444, Some(PixelFormat::Yuv444P)),
        stream_10bit(34, 18, ChromaFormat::Yuv420, Some(PixelFormat::Yuv420P10Le)),
        // 4:2:2 at an odd width: the chroma planes are 17 wide, where
        // `Yuv422P` at 35 wide has 18.
        stream_8bit(35, 19, ChromaFormat::Yuv422, None),
    ]
}

fn decoder() -> Box<dyn Decoder> {
    oxideav_dirac::decoder::make_decoder(&CodecParameters::video(CodecId::new("dirac")))
        .expect("decoder")
}

fn packet(bytes: &[u8]) -> Packet {
    Packet::new(0, TimeBase::new(1, 25), bytes.to_vec())
}

fn report(dec: &dyn Decoder) -> (Option<(u32, u32)>, Option<PixelFormat>) {
    (dec.output_video_dimensions(), dec.output_pixel_format())
}

/// Asserts the report and the planes of the frame just returned.
fn assert_frame(dec: &dyn Decoder, frame: &VideoFrame, want: &Stream, at: usize) {
    assert_eq!(
        report(dec),
        (Some(want.size), want.format),
        "report after frame {at}"
    );
    let planes = frame.image_planes();
    assert_eq!(planes.len(), 3, "frame {at}: planes");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = if i == 0 { want.size } else { want.chroma };
        assert_eq!(
            plane.stride,
            pw as usize * want.bytes_per_sample,
            "frame {at}: plane {i} stride"
        );
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "frame {at}: plane {i} rows"
        );
    }
}

fn receive(dec: &mut dyn Decoder, at: usize) -> VideoFrame {
    match dec.receive_frame() {
        Ok(Frame::Video(frame)) => frame,
        other => panic!("frame {at}: {other:?}"),
    }
}

/// All three streams sent before any frame is received: the first two
/// pictures wait behind newer sequence headers.
#[test]
fn each_frame_reports_its_own_size_and_layout() {
    let streams = streams();
    let mut dec = decoder();
    for s in &streams {
        dec.send_packet(&packet(&s.bytes)).expect("send");
    }
    // Before any frame, the next picture's own header, not the newest.
    assert_eq!(
        report(&*dec),
        (Some(streams[0].size), streams[0].format),
        "before the first frame"
    );
    for (at, want) in streams.iter().enumerate() {
        let frame = receive(&mut *dec, at);
        assert_frame(&*dec, &frame, want, at);
    }
    dec.flush().expect("flush");
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    // After the end, the last frame returned.
    assert_eq!(report(&*dec), (Some(streams[2].size), streams[2].format));
}

/// One stream per packet, receiving after each: a new sequence header
/// changes nothing until its picture is returned.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let streams = streams();
    let mut dec = decoder();
    for (at, want) in streams.iter().enumerate() {
        dec.send_packet(&packet(&want.bytes)).expect("send");
        if at > 0 {
            let before = &streams[at - 1];
            assert_eq!(
                report(&*dec),
                (Some(before.size), before.format),
                "frame {at}: before it is returned"
            );
        }
        let frame = receive(&mut *dec, at);
        assert_frame(&*dec, &frame, want, at);
        assert!(
            matches!(dec.receive_frame(), Err(Error::NeedMore)),
            "frame {at}: one picture"
        );
    }
}
