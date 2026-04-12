// Mexican Dollar example for individual types - see:
// Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
// https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf

use hypervector::example_mexican_dollar;
use hypervector::types::{
    binary::BinaryHDV, bipolar::BipolarHDV, complex::ComplexHDV, modular::ModularHDV, real::RealHDV,
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
    crate::example_mexican_dollar::<ComplexHDV<2048>>();
    println!("RealHDV");
    println!("=========");
    crate::example_mexican_dollar::<RealHDV<2048>>();
    println!("ModularHDV");
    println!("=========");
    crate::example_mexican_dollar::<ModularHDV<10000>>();
}
