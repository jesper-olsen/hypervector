// Mexican Dollar example for individual types - see:
// Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
// https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf

use hypervector::example_mexican_dollar;
use hypervector::types::{
    binary::Binary, bipolar::Bipolar, complex::ComplexHDV, modular::Modular, real::RealHDV,
};

fn main() {
    println!("Binary");
    println!("=========");
    crate::example_mexican_dollar::<Binary<10000>>();
    println!("Bipolar");
    println!("=========");
    crate::example_mexican_dollar::<Bipolar<10000>>();
    println!("ComplexHDV");
    println!("=========");
    crate::example_mexican_dollar::<ComplexHDV<2048>>();
    println!("RealHDV");
    println!("=========");
    crate::example_mexican_dollar::<RealHDV<2048>>();
    println!("Modular");
    println!("=========");
    crate::example_mexican_dollar::<Modular<10000>>();
}
