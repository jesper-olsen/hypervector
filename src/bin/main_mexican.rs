use hypervector::binary_hdv::BinaryHDV;
use hypervector::bipolar_hdv::BipolarHDV;
use hypervector::complex_hdv::ComplexHDV;
use hypervector::example_mexican_dollar;
use hypervector::modular_hdv::ModularHDV;
use hypervector::real_hdv::RealHDV;

fn main() {
    println!("Bipolar");
    example_mexican_dollar::<BipolarHDV<1024>>();
    println!("Binary");
    example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024
    println!("Real");
    example_mexican_dollar::<RealHDV<1024>>();
    println!("Complex");
    example_mexican_dollar::<ComplexHDV<1024>>();
    println!("Modular");
    example_mexican_dollar::<ModularHDV<1024>>();
}
