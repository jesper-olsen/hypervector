use hypervector::binary_hdv::BinaryHDV;
use hypervector::bipolar_hdv::BipolarHDV;
use hypervector::example_mexican_dollar;

fn main() {
    example_mexican_dollar::<BipolarHDV<1024>>();
    example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024
}
