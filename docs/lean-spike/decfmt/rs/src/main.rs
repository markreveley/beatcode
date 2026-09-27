use bc::decfmt::{format_dec, round_dec};
fn main() {
    let src = std::fs::read_to_string("inputs.txt").unwrap();
    // golden inputs
    let mut cases: Vec<(u64, u32)> = src.lines().map(|l| {
        let mut it = l.split_whitespace();
        (u64::from_str_radix(it.next().unwrap(), 16).unwrap(), it.next().unwrap().parse().unwrap())
    }).collect();
    // extra pseudo-random inputs: splitmix-ish over a few magnitudes
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for i in 0..40u64 {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^= z >> 31;
        let u = (z >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        let x = match i % 4 { 0 => u, 1 => u * 1000.0, 2 => -u * 27.0, _ => (u - 0.5) * 1e-3 };
        cases.push((x.to_bits(), [3u32, 6, 2, 0][(i % 4) as usize]));
    }
    for (b, n) in cases {
        let x = f64::from_bits(b);
        println!("({:#018X}, {}, {:#018X}, \"{}\"),", b, n, round_dec(x, n).to_bits(), format_dec(x, n));
    }
}
