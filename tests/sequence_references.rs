//! A new sequence predicts only from its own pictures. Picture numbers
//! restart with each sequence, so an inter picture of sequence B that
//! references picture 0 must find B's picture 0, not the previous
//! sequence's (here of another size). The previous sequence's references
//! are retired when B's first picture is decoded, not when B's header is
//! parsed: pictures of A still queued at that point keep theirs.
//!
//! Each sequence is an HQ intra reference picture 0 followed by an inter
//! picture 1 that references it, from the crate's own encoder.

use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, synthetic_translating_pair_64, InterEncoderParams,
    InterInputPicture,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// A frame's reported size and its planes.
type Decoded = ((u32, u32), Vec<Vec<u8>>);

fn intra_then_inter(
    size: u32,
    (y0, u0, v0): (&[u8], &[u8], &[u8]),
    (y1, u1, v1): (&[u8], &[u8], &[u8]),
) -> Vec<u8> {
    let seq = make_minimal_sequence(size, size, ChromaFormat::Yuv420);
    let intra = InterInputPicture {
        picture_number: 0,
        y: y0,
        u: u0,
        v: v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: y1,
        u: u1,
        v: v1,
    };
    encode_intra_then_inter_stream(
        &seq,
        &EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3),
        &InterEncoderParams::default(),
        &intra,
        &inter,
    )
}

/// Sequence A: a 32×32 ramp, then the ramp moved by two samples.
fn sequence_a() -> Vec<u8> {
    let ramp = |w: usize, shift: usize| -> Vec<u8> {
        (0..w * w)
            .map(|i| ((i % w + shift) * 7 % 256) as u8)
            .collect()
    };
    let (y0, y1) = (ramp(32, 0), ramp(32, 2));
    let c = vec![128u8; 16 * 16];
    intra_then_inter(32, (&y0, &c, &c), (&y1, &c, &c))
}

/// Sequence B: the crate's 64×64 translating test pair.
fn sequence_b() -> Vec<u8> {
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    intra_then_inter(64, (&y0, &u0, &v0), (&y1, &u1, &v1))
}

fn receive_all(dec: &mut dyn Decoder, out: &mut Vec<Decoded>) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let size = dec.output_video_dimensions().expect("reported size");
                out.push((size, frame.planes.into_iter().map(|p| p.data).collect()));
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// Decodes `streams` with one decoder, one packet each, receiving after
/// each packet when `receive_between` is set, else only at the end.
fn decode(streams: &[Vec<u8>], receive_between: bool) -> Vec<Decoded> {
    let mut dec =
        oxideav_dirac::decoder::make_decoder(&CodecParameters::video(CodecId::new("dirac")))
            .expect("decoder");
    let mut out = Vec::new();
    for stream in streams {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream.clone()))
            .expect("send");
        if receive_between {
            receive_all(&mut *dec, &mut out);
        }
    }
    dec.flush().expect("flush");
    receive_all(&mut *dec, &mut out);
    out
}

#[test]
fn a_new_sequence_predicts_from_its_own_pictures() {
    let (a, b) = (sequence_a(), sequence_b());
    let b_alone = decode(std::slice::from_ref(&b), true);
    assert_eq!(b_alone.len(), 2, "sequence B alone");
    let both = decode(&[a, b], true);
    assert_eq!(both.len(), 4, "frames");
    assert_eq!(both[2].0, (64, 64), "B's first frame");
    assert!(both[2..] == b_alone[..], "B after A differs from B alone");
}

#[test]
fn queued_sequences_keep_their_own_references() {
    let (a, b) = (sequence_a(), sequence_b());
    let a_alone = decode(std::slice::from_ref(&a), true);
    let b_alone = decode(std::slice::from_ref(&b), true);
    assert_eq!((a_alone.len(), b_alone.len()), (2, 2));
    // Both sequences parsed before any picture is decoded.
    let both = decode(&[a, b], false);
    assert_eq!(both.len(), 4, "frames");
    assert!(
        both[..2] == a_alone[..],
        "A queued before B differs from A alone"
    );
    assert!(
        both[2..] == b_alone[..],
        "B queued after A differs from B alone"
    );
}

/// A seek resets the decoder (FFmpeg's `dirac_decode_flush`): the stream
/// sent again from its start decodes as with a new decoder, though its
/// picture numbers are below the count the first pass reached. A seek
/// lands mid-stream, so no end of sequence comes before it: the stream is
/// sent without its own.
#[test]
fn a_reset_decoder_decodes_the_stream_again() {
    let mut b = sequence_b();
    let end = b.len() - 13;
    assert_eq!(
        (&b[end..end + 4], b[end + 4]),
        (&b"BBCD"[..], 0x10),
        "end of sequence"
    );
    b.truncate(end);
    let b_alone = decode(std::slice::from_ref(&b), true);
    assert_eq!(b_alone.len(), 2, "sequence B alone");
    let mut dec =
        oxideav_dirac::decoder::make_decoder(&CodecParameters::video(CodecId::new("dirac")))
            .expect("decoder");
    let mut passes = Vec::new();
    for _ in 0..2 {
        let mut out = Vec::new();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), b.clone()))
            .expect("send");
        receive_all(&mut *dec, &mut out);
        dec.reset().expect("reset");
        passes.push(out);
    }
    assert!(passes[0] == b_alone, "first pass differs from B alone");
    assert!(
        passes[1] == b_alone,
        "the pass after the reset differs from B alone"
    );
}
