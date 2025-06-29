use hypervector::binary_hdv::BinaryHDV;
use hypervector::bipolar_hdv::BipolarHDV;
use hypervector::example_mexican_dollar;

fn test_accumulate() {
    let mut v1 = BinaryHDV::<2>::zero();
    let mut v2 = BinaryHDV::<2>::zero();
    v1.data[0] = 5;
    v1.data[1] = 1;
    v2.data[0] = 1;
    v2.data[1] = 0;
    let b = BinaryHDV::<2>::acc(&[&v1, &v2]);
    println!("0: {:b} 1: {:b}", b.data[0], b.data[1]);
}
fn main() {
    test_accumulate();
    example_mexican_dollar::<BipolarHDV<1000>>();
    example_mexican_dollar::<BinaryHDV<16>>();
}
