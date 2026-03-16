use hypervector::{HyperVector, binary_hdv::BinaryHDV, hdv};
use mersenne_twister_rs::MersenneTwister64;

hdv!(binary, BinaryHDV1024, 1024);

fn main() {
    let mut rng = MersenneTwister64::new(42);
    let h1 = BinaryHDV1024::random(&mut rng);
    //let h1 = BinaryHDV::<16>::random(&mut rng);
    let h2 = BinaryHDV::<16>::random(&mut rng);
    let h3 = BinaryHDV::<16>::bundle(&[&h1, &h2]);
    let width = 200;
    print!("h1: {}", h1.to_braille(width));
    print!("h3: {}", h3.to_braille(width));
    print!("df: {}", h1.diff_braille(&h3, width));
}
