use bc::decfmt::{format_dec, round_dec};
fn main() {
    for n in [2u32, 3, 6] {
        let d = 9007199254740992.0f64 / 10f64.powi(n as i32);
        for x in [d, f64::from_bits(d.to_bits() - 1), f64::from_bits(d.to_bits() + 1)] {
            println!("n={n} x={x:?} bits={:#018X} round_dec={:?} same={} fmt={}", x.to_bits(), round_dec(x, n), round_dec(x, n).to_bits() == x.to_bits(), format_dec(x, n));
        }
    }
}
