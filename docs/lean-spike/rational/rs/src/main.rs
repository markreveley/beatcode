use bc::rational::Rational;
fn show(r: Result<Rational, bc::rational::RatError>) -> String { match r { Ok(q) => format!("ok {}", q.to_s()), Err(e) => format!("err {:?}", e) } }
fn main() {
  println!("new 0 1 => {}", show(Rational::new(0i64, 1i64)));
  println!("new 0 -7 => {}", show(Rational::new(0i64, -7i64)));
  println!("new 0 5 => {}", show(Rational::new(0i64, 5i64)));
  println!("new 1 2 => {}", show(Rational::new(1i64, 2i64)));
  println!("new 2 4 => {}", show(Rational::new(2i64, 4i64)));
  println!("new -1 4 => {}", show(Rational::new(-1i64, 4i64)));
  println!("new 1 -4 => {}", show(Rational::new(1i64, -4i64)));
  println!("new -1 -4 => {}", show(Rational::new(-1i64, -4i64)));
  println!("new 6 -9 => {}", show(Rational::new(6i64, -9i64)));
  println!("new -6 -9 => {}", show(Rational::new(-6i64, -9i64)));
  println!("new 100 -1 => {}", show(Rational::new(100i64, -1i64)));
  println!("new 9223372036854775807 1 => {}", show(Rational::new(9223372036854775807i64, 1i64)));
  println!("new -9223372036854775808 1 => {}", show(Rational::new(-9223372036854775808i64, 1i64)));
  println!("new -9223372036854775808 -1 => {}", show(Rational::new(-9223372036854775808i64, -1i64)));
  println!("new 9223372036854775807 -1 => {}", show(Rational::new(9223372036854775807i64, -1i64)));
  println!("new -9223372036854775808 -9223372036854775808 => {}", show(Rational::new(-9223372036854775808i64, -9223372036854775808i64)));
  println!("new -9223372036854775808 2 => {}", show(Rational::new(-9223372036854775808i64, 2i64)));
  println!("new -9223372036854775808 -2 => {}", show(Rational::new(-9223372036854775808i64, -2i64)));
  println!("new 9223372036854775807 9223372036854775807 => {}", show(Rational::new(9223372036854775807i64, 9223372036854775807i64)));
  println!("new 9223372036854775807 -9223372036854775808 => {}", show(Rational::new(9223372036854775807i64, -9223372036854775808i64)));
  println!("new 1 0 => {}", show(Rational::new(1i64, 0i64)));
  println!("new 0 0 => {}", show(Rational::new(0i64, 0i64)));
  println!("new -9223372036854775808 0 => {}", show(Rational::new(-9223372036854775808i64, 0i64)));
  println!("new 9223372036854775807 3 => {}", show(Rational::new(9223372036854775807i64, 3i64)));
  println!("new 1 -9223372036854775808 => {}", show(Rational::new(1i64, -9223372036854775808i64)));
  println!("new 2 -9223372036854775808 => {}", show(Rational::new(2i64, -9223372036854775808i64)));
  println!("new -2 -9223372036854775808 => {}", show(Rational::new(-2i64, -9223372036854775808i64)));
  println!("new 9223372036854775807 9223372036854775806 => {}", show(Rational::new(9223372036854775807i64, 9223372036854775806i64)));
  println!("new -9223372036854775808 9223372036854775807 => {}", show(Rational::new(-9223372036854775808i64, 9223372036854775807i64)));
  println!("new 0 -9223372036854775808 => {}", show(Rational::new(0i64, -9223372036854775808i64)));
  println!("new 12 18 => {}", show(Rational::new(12i64, 18i64)));
  println!("new -12 18 => {}", show(Rational::new(-12i64, 18i64)));
  println!("add 1/2 1/3 => {}", show(Rational::new(1i64,2i64).unwrap().add(Rational::new(1i64,3i64).unwrap())));
  println!("add 1/2 -1/2 => {}", show(Rational::new(1i64,2i64).unwrap().add(Rational::new(-1i64,2i64).unwrap())));
  println!("add -1/4 1/4 => {}", show(Rational::new(-1i64,4i64).unwrap().add(Rational::new(1i64,4i64).unwrap())));
  println!("add 1/3 2/3 => {}", show(Rational::new(1i64,3i64).unwrap().add(Rational::new(2i64,3i64).unwrap())));
  println!("add 1/3 1/6 => {}", show(Rational::new(1i64,3i64).unwrap().add(Rational::new(1i64,6i64).unwrap())));
  println!("add 9223372036854775807/1 1/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().add(Rational::new(1i64,1i64).unwrap())));
  println!("add 9223372036854775807/1 -9223372036854775807/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().add(Rational::new(-9223372036854775807i64,1i64).unwrap())));
  println!("add 9223372036854775807/2 9223372036854775807/2 => {}", show(Rational::new(9223372036854775807i64,2i64).unwrap().add(Rational::new(9223372036854775807i64,2i64).unwrap())));
  println!("add 9223372036854775807/1 9223372036854775807/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().add(Rational::new(9223372036854775807i64,1i64).unwrap())));
  println!("add -9223372036854775808/1 -1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().add(Rational::new(-1i64,1i64).unwrap())));
  println!("add -9223372036854775808/1 -9223372036854775808/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().add(Rational::new(-9223372036854775808i64,1i64).unwrap())));
  println!("add 1/9223372036854775807 1/9223372036854775807 => {}", show(Rational::new(1i64,9223372036854775807i64).unwrap().add(Rational::new(1i64,9223372036854775807i64).unwrap())));
  println!("add 1/9223372036854775807 1/9223372036854775806 => {}", show(Rational::new(1i64,9223372036854775807i64).unwrap().add(Rational::new(1i64,9223372036854775806i64).unwrap())));
  println!("add 1/4611686018427387904 1/4611686018427387904 => {}", show(Rational::new(1i64,4611686018427387904i64).unwrap().add(Rational::new(1i64,4611686018427387904i64).unwrap())));
  println!("add -3/7 -4/7 => {}", show(Rational::new(-3i64,7i64).unwrap().add(Rational::new(-4i64,7i64).unwrap())));
  println!("add 0/1 5/-7 => {}", show(Rational::new(0i64,1i64).unwrap().add(Rational::new(5i64,-7i64).unwrap())));
  println!("add -9223372036854775808/1 0/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().add(Rational::new(0i64,1i64).unwrap())));
  println!("add 9223372036854775807/1 0/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().add(Rational::new(0i64,1i64).unwrap())));
  println!("add -9223372036854775808/4611686018427387904 -9223372036854775808/4611686018427387904 => {}", show(Rational::new(-9223372036854775808i64,4611686018427387904i64).unwrap().add(Rational::new(-9223372036854775808i64,4611686018427387904i64).unwrap())));
  println!("add -9223372036854775808/1 1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().add(Rational::new(1i64,1i64).unwrap())));
  println!("add 9223372036854775807/1 -1/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().add(Rational::new(-1i64,1i64).unwrap())));
  println!("add -9223372036854775808/2 -9223372036854775808/2 => {}", show(Rational::new(-9223372036854775808i64,2i64).unwrap().add(Rational::new(-9223372036854775808i64,2i64).unwrap())));
  println!("mul 1/2 2/3 => {}", show(Rational::new(1i64,2i64).unwrap().mul(Rational::new(2i64,3i64).unwrap())));
  println!("mul -1/2 -2/3 => {}", show(Rational::new(-1i64,2i64).unwrap().mul(Rational::new(-2i64,3i64).unwrap())));
  println!("mul -1/2 2/3 => {}", show(Rational::new(-1i64,2i64).unwrap().mul(Rational::new(2i64,3i64).unwrap())));
  println!("mul 9223372036854775807/1 1/9223372036854775807 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().mul(Rational::new(1i64,9223372036854775807i64).unwrap())));
  println!("mul 9223372036854775807/1 2/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().mul(Rational::new(2i64,1i64).unwrap())));
  println!("mul -9223372036854775808/1 -1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().mul(Rational::new(-1i64,1i64).unwrap())));
  println!("mul -9223372036854775808/1 1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().mul(Rational::new(1i64,1i64).unwrap())));
  println!("mul -9223372036854775808/1 1/2 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().mul(Rational::new(1i64,2i64).unwrap())));
  println!("mul -9223372036854775808/1 0/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().mul(Rational::new(0i64,1i64).unwrap())));
  println!("mul 4294967296/1 2147483648/1 => {}", show(Rational::new(4294967296i64,1i64).unwrap().mul(Rational::new(2147483648i64,1i64).unwrap())));
  println!("mul 4294967296/1 -2147483648/1 => {}", show(Rational::new(4294967296i64,1i64).unwrap().mul(Rational::new(-2147483648i64,1i64).unwrap())));
  println!("mul 1/4294967296 1/2147483648 => {}", show(Rational::new(1i64,4294967296i64).unwrap().mul(Rational::new(1i64,2147483648i64).unwrap())));
  println!("mul 1/4294967296 1/2147483647 => {}", show(Rational::new(1i64,4294967296i64).unwrap().mul(Rational::new(1i64,2147483647i64).unwrap())));
  println!("mul 3/4 4/3 => {}", show(Rational::new(3i64,4i64).unwrap().mul(Rational::new(4i64,3i64).unwrap())));
  println!("mul 9223372036854775807/9223372036854775806 9223372036854775806/9223372036854775807 => {}", show(Rational::new(9223372036854775807i64,9223372036854775806i64).unwrap().mul(Rational::new(9223372036854775806i64,9223372036854775807i64).unwrap())));
  println!("mul 9223372036854775807/2 2/9223372036854775807 => {}", show(Rational::new(9223372036854775807i64,2i64).unwrap().mul(Rational::new(2i64,9223372036854775807i64).unwrap())));
  println!("mul -9223372036854775808/3 3/-4611686018427387904 => {}", show(Rational::new(-9223372036854775808i64,3i64).unwrap().mul(Rational::new(3i64,-4611686018427387904i64).unwrap())));
  println!("mul 0/-5 9223372036854775807/1 => {}", show(Rational::new(0i64,-5i64).unwrap().mul(Rational::new(9223372036854775807i64,1i64).unwrap())));
  println!("mul -9223372036854775808/1 -1/2 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().mul(Rational::new(-1i64,2i64).unwrap())));
  println!("divr 1/2 1/3 => {}", show(Rational::new(1i64,2i64).unwrap().divr(Rational::new(1i64,3i64).unwrap())));
  println!("divr 1/2 0/1 => {}", show(Rational::new(1i64,2i64).unwrap().divr(Rational::new(0i64,1i64).unwrap())));
  println!("divr 0/1 0/1 => {}", show(Rational::new(0i64,1i64).unwrap().divr(Rational::new(0i64,1i64).unwrap())));
  println!("divr 0/1 5/1 => {}", show(Rational::new(0i64,1i64).unwrap().divr(Rational::new(5i64,1i64).unwrap())));
  println!("divr 1/2 -1/3 => {}", show(Rational::new(1i64,2i64).unwrap().divr(Rational::new(-1i64,3i64).unwrap())));
  println!("divr -1/2 -1/3 => {}", show(Rational::new(-1i64,2i64).unwrap().divr(Rational::new(-1i64,3i64).unwrap())));
  println!("divr 1/1 -9223372036854775808/1 => {}", show(Rational::new(1i64,1i64).unwrap().divr(Rational::new(-9223372036854775808i64,1i64).unwrap())));
  println!("divr 1/1 9223372036854775807/1 => {}", show(Rational::new(1i64,1i64).unwrap().divr(Rational::new(9223372036854775807i64,1i64).unwrap())));
  println!("divr -9223372036854775808/1 -1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().divr(Rational::new(-1i64,1i64).unwrap())));
  println!("divr -9223372036854775808/1 1/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().divr(Rational::new(1i64,1i64).unwrap())));
  println!("divr -9223372036854775808/1 -9223372036854775808/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().divr(Rational::new(-9223372036854775808i64,1i64).unwrap())));
  println!("divr 9223372036854775807/1 9223372036854775807/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().divr(Rational::new(9223372036854775807i64,1i64).unwrap())));
  println!("divr 9223372036854775807/1 1/9223372036854775807 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().divr(Rational::new(1i64,9223372036854775807i64).unwrap())));
  println!("divr 1/9223372036854775807 9223372036854775807/1 => {}", show(Rational::new(1i64,9223372036854775807i64).unwrap().divr(Rational::new(9223372036854775807i64,1i64).unwrap())));
  println!("divr -9223372036854775808/1 2/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().divr(Rational::new(2i64,1i64).unwrap())));
  println!("divr -9223372036854775808/1 -2/1 => {}", show(Rational::new(-9223372036854775808i64,1i64).unwrap().divr(Rational::new(-2i64,1i64).unwrap())));
  println!("divr 2/3 4/9 => {}", show(Rational::new(2i64,3i64).unwrap().divr(Rational::new(4i64,9i64).unwrap())));
  println!("divr 1/3 -9223372036854775808/1 => {}", show(Rational::new(1i64,3i64).unwrap().divr(Rational::new(-9223372036854775808i64,1i64).unwrap())));
  println!("divr -1/4 1/4 => {}", show(Rational::new(-1i64,4i64).unwrap().divr(Rational::new(1i64,4i64).unwrap())));
  println!("divr 1/1 -1/-4611686018427387904 => {}", show(Rational::new(1i64,1i64).unwrap().divr(Rational::new(-1i64,-4611686018427387904i64).unwrap())));
  println!("divr 9223372036854775807/1 -1/1 => {}", show(Rational::new(9223372036854775807i64,1i64).unwrap().divr(Rational::new(-1i64,1i64).unwrap())));
  { let q = Rational::new(-1i64,4i64).unwrap(); println!("un -1 4 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(7i64,2i64).unwrap(); println!("un 7 2 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(-7i64,2i64).unwrap(); println!("un -7 2 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(1i64,1i64).unwrap(); println!("un 1 1 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(0i64,-3i64).unwrap(); println!("un 0 -3 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(-9223372036854775808i64,1i64).unwrap(); println!("un -9223372036854775808 1 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(9223372036854775807i64,1i64).unwrap(); println!("un 9223372036854775807 1 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(-9223372036854775808i64,2i64).unwrap(); println!("un -9223372036854775808 2 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(-9223372036854775807i64,2i64).unwrap(); println!("un -9223372036854775807 2 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(5i64,-2i64).unwrap(); println!("un 5 -2 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(1i64,3i64).unwrap(); println!("un 1 3 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(9223372036854775807i64,9223372036854775806i64).unwrap(); println!("un 9223372036854775807 9223372036854775806 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(1i64,-4611686018427387904i64).unwrap(); println!("un 1 -4611686018427387904 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
  { let q = Rational::new(3i64,3074457345618258602i64).unwrap(); println!("un 3 3074457345618258602 => floor={} int={} s={} f=0x{:016x}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }
}
