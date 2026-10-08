// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg 2da55bf libavcodec/diracdec.c: the reference choice and
// retirement of dirac_decode_picture_header (MAX_REFERENCE_FRAMES), and the
// picture output order of dirac_decode_frame and get_delayed_pic
// (remove_frame, add_frame, MAX_DELAY).
// Copyright (C) 2007 Marco Gerards <marco@gnu.org>
// Copyright (C) 2009 David Conrad
// Copyright (C) 2011 Jordi Ortiz
//
// This file is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1, or (at your option) any later version.
// It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE-LGPL.

//! FFmpeg's choice of the references a picture predicts from, of the
//! references it keeps, and of the order pictures are shown in.

use crate::picture::ReferencePicture;
use crate::sequence::SequenceHeader;

/// FFmpeg's `MAX_REFERENCE_FRAMES`.
const MAX_REFERENCE_FRAMES: usize = 8;
/// FFmpeg's `MAX_DELAY`: decoded pictures held for output order.
const MAX_DELAY: usize = 5;

/// The held reference whose picture number is closest to `n`, the first
/// such in buffer order: a picture naming one that never arrived still
/// predicts from its nearest neighbour.
pub(crate) fn closest_reference(refs: &[ReferencePicture], n: u32) -> Option<&ReferencePicture> {
    let mut best: Option<(&ReferencePicture, i64)> = None;
    for r in refs {
        let dist = (i64::from(r.picture_number) - i64::from(n)).abs();
        if best.map_or(true, |(_, d)| dist < d) {
            best = Some((r, dist));
            if dist == 0 {
                break;
            }
        }
    }
    best.map(|(r, _)| r)
}

/// The reference FFmpeg predicts from when it holds none: a new frame
/// buffer, all samples 0 (its buffer pool zeroes a new one), which is
/// `-2^(depth-1)` in the pre-output-offset domain references are held in.
pub(crate) fn blank_reference(seq: &SequenceHeader) -> ReferencePicture {
    let half = |depth: u32| if depth == 0 { 0 } else { 1i32 << (depth - 1) };
    let (lw, lh) = (seq.luma_width as usize, seq.luma_height as usize);
    let (cw, ch) = (seq.chroma_width as usize, seq.chroma_height as usize);
    ReferencePicture {
        picture_number: 0,
        luma_width: lw,
        luma_height: lh,
        chroma_width: cw,
        chroma_height: ch,
        y: vec![-half(seq.luma_depth); lw * lh],
        u: vec![-half(seq.chroma_depth); cw * ch],
        v: vec![-half(seq.chroma_depth); cw * ch],
    }
}

/// Keep a decoded reference picture: first drop the reference its header
/// retires, then add it; with `MAX_REFERENCE_FRAMES` held, drop the oldest.
/// `refs` is oldest first.
pub(crate) fn admit_reference(
    refs: &mut Vec<ReferencePicture>,
    retired: Option<u32>,
    picture: ReferencePicture,
) {
    if let Some(at) = retired.and_then(|n| refs.iter().rposition(|r| r.picture_number == n)) {
        refs.remove(at);
    }
    if refs.len() == MAX_REFERENCE_FRAMES {
        refs.remove(0);
    }
    refs.push(picture);
}

/// FFmpeg's output order over decoded pictures `T`: its `frame_number` and
/// `delay_frames`.
pub(crate) struct OutputOrder<T> {
    /// The picture number to show next, set by the first picture decoded.
    next: Option<i64>,
    /// Pictures waiting for their turn, with their numbers, in arrival
    /// order.
    waiting: Vec<(u32, T)>,
}

impl<T> OutputOrder<T> {
    pub(crate) const fn new() -> Self {
        Self {
            next: None,
            waiting: Vec::new(),
        }
    }

    /// The picture to show now, if any, once picture `number` is decoded.
    /// A picture ahead of the next number waits, and the waiting picture
    /// with that number goes out; with `MAX_DELAY` waiting, the
    /// lowest-numbered one goes out instead. The picture with that number
    /// goes straight out; one behind it is dropped.
    pub(crate) fn push(&mut self, number: u32, picture: T) -> Option<T> {
        let n = i64::from(number);
        let next = *self.next.get_or_insert(n);
        if n > next {
            let mut out = self.remove_last(next);
            if self.waiting.len() == MAX_DELAY {
                let lowest = self.waiting.iter().map(|(k, _)| *k).min();
                out = lowest.and_then(|k| self.remove_last(i64::from(k)));
            }
            self.waiting.push((number, picture));
            let (k, out) = out?;
            self.next = Some(i64::from(k) + 1);
            Some(out)
        } else if n == next {
            self.next = Some(n + 1);
            Some(picture)
        } else {
            None
        }
    }

    /// The first lowest-numbered waiting picture: what goes out at the end
    /// of the stream (get_delayed_pic).
    pub(crate) fn take_lowest(&mut self) -> Option<T> {
        let at = self
            .waiting
            .iter()
            .enumerate()
            .min_by_key(|(_, (k, _))| *k)?
            .0;
        Some(self.waiting.remove(at).1)
    }

    /// Nothing waiting and no number to show next (dirac_decode_flush).
    pub(crate) fn clear(&mut self) {
        self.next = None;
        self.waiting.clear();
    }

    /// The last waiting picture numbered `number` (remove_frame).
    fn remove_last(&mut self, number: i64) -> Option<(u32, T)> {
        let at = self
            .waiting
            .iter()
            .rposition(|(k, _)| i64::from(*k) == number)?;
        Some(self.waiting.remove(at))
    }
}
