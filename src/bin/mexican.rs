// Mexican Dollar example for individual types - see:
// Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
// https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf

use hypervector::{
    binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV, complex_hdv::ComplexHDV,
    example_mexican_dollar, example_mexican_dollar2, modular_hdv::ModularHDV, real_hdv::RealHDV,
};

fn main() {
    println!("BinaryHDV");
    println!("=========");
    crate::example_mexican_dollar::<BinaryHDV<10000>>();
    println!("BipolarHDV");
    println!("=========");
    crate::example_mexican_dollar::<BipolarHDV<10000>>();
    println!("ComplexHDV");
    println!("=========");
    crate::example_mexican_dollar2::<ComplexHDV<2048>>();
    println!("RealHDV");
    println!("=========");
    crate::example_mexican_dollar2::<RealHDV<2048>>();
    println!("ModularHDV");
    println!("=========");
    crate::example_mexican_dollar2::<ModularHDV<10000>>();
}
