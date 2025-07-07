use hypervector::binary_hdv::BinaryHDV;
use hypervector::bipolar_hdv::BipolarHDV;
use hypervector::complex_hdv::ComplexHDV;
use hypervector::example_mexican_dollar;
use hypervector::real_hdv::RealHDV;

fn main() {
    example_mexican_dollar::<BipolarHDV<1024>>();
    example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024
    example_mexican_dollar::<RealHDV<1024>>();
    example_mexican_dollar::<ComplexHDV<1024>>();
}
