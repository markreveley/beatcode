use bc::prng::*;
fn main(){
  println!("fnv 0 {}", fnv(""));
  println!("fnv 1 {}", fnv("a"));
  println!("fnv 2 {}", fnv("ab"));
  println!("fnv 3 {}", fnv("abc"));
  println!("fnv 4 {}", fnv("kick"));
  println!("fnv 5 {}", fnv("snare"));
  println!("fnv 6 {}", fnv("hat"));
  println!("fnv 7 {}", fnv("clap"));
  println!("fnv 8 {}", fnv("sample|kick"));
  println!("fnv 9 {}", fnv("sample|kick-click"));
  println!("fnv 10 {}", fnv("sample|snare"));
  println!("fnv 11 {}", fnv("sample|hat"));
  println!("fnv 12 {}", fnv("sample|clap"));
  println!("fnv 13 {}", fnv("0|1"));
  println!("fnv 14 {}", fnv("hum|123"));
  println!("fnv 15 {}", fnv("kick|1"));
  println!("fnv 16 {}", fnv("prob|0"));
  println!("fnv 17 {}", fnv("x|18446744073709551615"));
  println!("fnv 18 {}", fnv("é"));
  { let seed = mask_seed(1i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 19 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 20 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 21 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 22 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 23 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 24 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 25 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(1i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 26 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 27 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 28 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 29 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 30 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 31 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 32 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 33 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(2i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 34 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 35 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 36 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 37 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 38 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 39 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 40 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 41 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(7i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 42 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 43 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 44 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 45 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 46 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 47 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 48 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 49 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(13i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 50 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 51 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 52 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 53 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 54 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 55 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 56 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 57 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(41i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 58 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 59 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 60 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 61 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 62 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 63 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 64 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 65 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(-1i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 66 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 67 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 68 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 69 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 70 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 71 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 72 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 73 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(18446744073709551615i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 74 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 75 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Str("kick"),Part::Str("hum"),Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 76 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Str("kick"),Part::Str("prob"),Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 77 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Str("hat"),Part::Str("prob"),Part::Int(31)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 78 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Str("snare"),Part::Str("hum"),Part::Int(7)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 79 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Int(0)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 80 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Int(1)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 81 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let seed = mask_seed(12345678901234567890i128); let parts=[Part::Int(41)]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt 82 {} {} {} {:016x}", seed, k, sm, f.to_bits()); }
  { let base = fnv(&format!("sample|{}", "kick-click")); let v = noise("kick-click", 8); for j in 0..8 { let k=flt_key(base,&[Part::Int(j as i64)]); let sm=splitmix64(k); println!("noise 83 {} {} {} {} {:016x}", j, base, k, sm, v[j].to_bits()); } }
  { let base = fnv(&format!("sample|{}", "snare")); let v = noise("snare", 8); for j in 0..8 { let k=flt_key(base,&[Part::Int(j as i64)]); let sm=splitmix64(k); println!("noise 84 {} {} {} {} {:016x}", j, base, k, sm, v[j].to_bits()); } }
  { let base = fnv(&format!("sample|{}", "hat")); let v = noise("hat", 8); for j in 0..8 { let k=flt_key(base,&[Part::Int(j as i64)]); let sm=splitmix64(k); println!("noise 85 {} {} {} {} {:016x}", j, base, k, sm, v[j].to_bits()); } }
  { let base = fnv(&format!("sample|{}", "clap")); let v = noise("clap", 8); for j in 0..8 { let k=flt_key(base,&[Part::Int(j as i64)]); let sm=splitmix64(k); println!("noise 86 {} {} {} {} {:016x}", j, base, k, sm, v[j].to_bits()); } }
}
