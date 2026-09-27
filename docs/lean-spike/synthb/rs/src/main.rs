use bc::events::Kind;
use bc::score::SampleKind;
use bc::synth::{Kit, note_freq, sin_p};

fn sin_inputs() -> Vec<f64> {
    let mut v = Vec::new();
    // 181 points spanning [-100, 100] in steps of 100/90 (not dyadic on purpose)
    for i in 0..=180 {
        v.push(-100.0 + (i as f64) * (100.0 / 90.0));
    }
    // 19 extra: huge / special / near-boundary values
    let extra = [
        0.0, -0.0, 1e-300, 1.5707963267948966, 3.141592653589793,
        6.283185307179586, 1e6, 1e10, 1e15, 1e16, 1e17, 1e20, 1e300,
        -1e300, 4503599627370496.0, 9007199254740992.0,
        0.7853981633974483, 123456.789, -98765.4321,
    ];
    v.extend_from_slice(&extra);
    v
}

fn main() {
    // 1. sin_p bits
    println!("SIN");
    for x in sin_inputs() {
        println!("{:016x} {:016x}", x.to_bits(), sin_p(x).to_bits());
    }
    // 2. note_freq bits for midi 0..127
    println!("NOTE");
    for m in 0..128i64 {
        println!("{} {:016x}", m, note_freq(m).to_bits());
    }
    // 3. kick / hat buffers via the public Kit
    let mut kit = Kit::new();
    let kick = kit.buffer(Kind::Sample(SampleKind::Kick), None);
    let hat = kit.buffer(Kind::Sample(SampleKind::Hat), None);
    let bytes = |b: &[f64]| -> Vec<u8> { b.iter().flat_map(|x| x.to_le_bytes()).collect() };
    println!("KICK len={} sha256={}", kick.len(), bc::sha256::hex(&bytes(&kick)));
    println!("HAT len={} sha256={}", hat.len(), bc::sha256::hex(&bytes(&hat)));
    // Also dump full bit tables for diffing
    println!("KICKBITS");
    for x in kick.iter() { println!("{:016x}", x.to_bits()); }
    println!("HATBITS");
    for x in hat.iter() { println!("{:016x}", x.to_bits()); }
}
