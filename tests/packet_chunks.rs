//! Container packets may group or split Layer II frames. Preserve every
//! sample and the synthesis history; wait for EOF before padding a cut tail.
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_mp2::{codec_decoder::make_decoder, header::FrameHeader};

const DATA: &[u8] = include_bytes!("fixtures/stereo_48k_192.mp2");
fn decoder() -> Box<dyn Decoder> {
    make_decoder(&CodecParameters::audio(CodecId::new("mp2"))).unwrap()
}
fn packet(bytes: &[u8]) -> Packet { Packet::new(0, TimeBase::new(1, 48000), bytes.to_vec()) }
fn drain(decoder: &mut dyn Decoder, out: &mut Vec<i16>) -> usize {
    let mut count = 0;
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Audio(frame)) => {
                assert_eq!(frame.samples, 1152);
                assert_eq!(frame.data.len(), 2);
                for i in 0..frame.samples as usize {
                    for plane in &frame.data {
                        out.push(i16::from_le_bytes([plane[2 * i], plane[2 * i + 1]]));
                    }
                }
                count += 1;
            }
            Ok(_) => panic!("not audio"),
            Err(Error::NeedMore | Error::Eof) => return count,
            Err(err) => panic!("{err}"),
        }
    }
}

#[test]
fn grouped_and_split_frames_preserve_all_pcm_and_reset_state() {
    let size = FrameHeader::parse(DATA).unwrap().frame_size_bytes();
    assert_eq!(DATA.len() % size, 0);
    let mut expected = Vec::new();
    let mut dec = decoder();
    for frame in DATA.chunks(size) {
        dec.send_packet(&packet(frame)).unwrap();
        drain(dec.as_mut(), &mut expected);
    }
    assert_eq!(expected.len(), DATA.len() / size * 1152 * 2);
    for chunk in [DATA.len(), 2304, 997, 3, 1] {
        dec.reset().unwrap();
        let mut actual = Vec::new();
        for bytes in DATA.chunks(chunk) {
            dec.send_packet(&packet(bytes)).unwrap();
            drain(dec.as_mut(), &mut actual);
        }
        dec.flush().unwrap();
        drain(dec.as_mut(), &mut actual);
        assert_eq!(actual, expected, "packet bytes={chunk}");
    }
}

#[test]
fn an_incomplete_frame_waits_and_the_final_tail_is_padded_once() {
    let size = FrameHeader::parse(DATA).unwrap().frame_size_bytes();
    let cut = size + size / 2;
    let mut dec = decoder();
    dec.send_packet(&packet(&DATA[..cut])).unwrap();
    let mut pcm = Vec::new();
    assert_eq!(drain(dec.as_mut(), &mut pcm), 1);
    dec.flush().unwrap();
    assert_eq!(drain(dec.as_mut(), &mut pcm), 1);
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    let mut padded = DATA[..cut].to_vec();
    padded.resize(size * 2, 0);
    dec.reset().unwrap();
    dec.send_packet(&packet(&padded)).unwrap();
    let mut expected = Vec::new();
    assert_eq!(drain(dec.as_mut(), &mut expected), 2);
    assert_eq!(pcm, expected);
    // Reset must discard a buffered partial frame, not prepend it after seek.
    dec.reset().unwrap();
    dec.send_packet(&packet(&DATA[..3])).unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    dec.reset().unwrap();
    dec.send_packet(&packet(&padded)).unwrap();
    let mut reset = Vec::new();
    drain(dec.as_mut(), &mut reset);
    assert_eq!(reset, expected);
}

#[test]
fn one_timestamp_per_packet_and_lazy_bounded_input() {
    let size = FrameHeader::parse(DATA).unwrap().frame_size_bytes();
    let mut dec = decoder();
    dec.send_packet(&packet(&DATA[..size * 2]).with_pts(900)).unwrap();
    let Frame::Audio(first) = dec.receive_frame().unwrap() else { panic!("audio") };
    let Frame::Audio(second) = dec.receive_frame().unwrap() else { panic!("audio") };
    assert_eq!(first.pts, Some(900));
    assert_eq!(second.pts, None, "do not stamp both frames with the same time");
    dec.reset().unwrap();
    let large = vec![0xff; 8 * 1024 * 1024 + 1];
    assert!(dec.send_packet(&packet(&large)).is_err());
    dec.send_packet(&packet(&DATA[..size])).unwrap();
    let Frame::Audio(frame) = dec.receive_frame().unwrap() else { panic!("audio") };
    assert_eq!(frame.data, first.data, "rejected input must not alter synthesis state");
}
