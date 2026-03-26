// Associative memory test
// Given a dimension: 
// 1. Generate N key value pairs.   
// 2. Bind keys to values
// 3. Bundle all
// 4. For each key, unbind from bundle and try to identify matching value

use hypervector::{
    HyperVector, UnitAccumulator, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV, cleanup,
    complex_hdv::ComplexHDV, hdv, modular_hdv::ModularHDV, nearest, real_hdv::RealHDV,
    save_hypervectors_to_csv,
};
use mersenne_twister_rs::MersenneTwister64;
use std::io;

hdv!(binary, BinaryHDV1024, 1024); //   1024 bits
hdv!(binary, BinaryHDV8192, 8192);
hdv!(binary, BinaryHDV65536, 65536);
hdv!(modular, ModularHDV1024, 1024); //   8192 bits
hdv!(modular, ModularHDV8192, 8192); //  65536 bits
hdv!(modular, ModularHDV16384, 16384); // 131072 bits
hdv!(real, RealHDV1024, 1024); //  65536 bits
hdv!(real, RealHDV2048, 2048); // 131072 bits
hdv!(complex, ComplexHDV1024, 1024); // 131072 bits

fn test_suite(label: &str, f: fn(usize) -> Result<(), io::Error>) -> Result<(), io::Error> {
    println!("--- {label} ---");
    f(10)?;
    for n in (100..=500).step_by(100) {
        f(n)?;
    }
    Ok(())
}

fn run<T: HyperVector + Clone>(n: usize) -> Result<(), io::Error> {
    let mut rng = MersenneTwister64::new(42);
    let keys: Vec<T> = (0..n).map(|_| T::random(&mut rng)).collect();
    let values: Vec<T> = (0..n).map(|_| T::random(&mut rng)).collect();
    let mut acc = T::UnitAccumulator::new();
    for (k, v) in keys.iter().zip(values.iter()) {
        let kv = k.bind(v);
        acc.add(&kv)
    }
    let m = acc.finalize();
    let correct: usize = (0..n)
        .map(|i| {
            let v = m.unbind(&keys[i]);
            let (idx, _) = nearest(&v, &values);
            if i == idx { 1 } else { 0 }
        })
        .sum();
    let acc = 100.0 * correct as f64 / n as f64;
    println!("Accuracy: {correct}/{n} = {acc:.2}%");
    Ok(())
}

fn main() -> Result<(), io::Error> {
    let groups: &[(&str, &[(&str, fn(usize) -> Result<(), io::Error>)])] = &[
        ("8192 bits", &[
            ("Binary  DIM  8192", run::<BinaryHDV8192>),
            ("Modular DIM  1024", run::<ModularHDV1024>),
        ]),
        ("65536 bits", &[
            ("Binary  DIM 65536", run::<BinaryHDV65536>),
            ("Modular DIM  8192", run::<ModularHDV8192>),
            ("Real    DIM  1024", run::<RealHDV1024>),
        ]),
        ("131072 bits", &[
            ("Modular DIM 16384", run::<ModularHDV16384>),
            ("Real    DIM  2048", run::<RealHDV2048>),
            ("Complex DIM  1024", run::<ComplexHDV1024>),
        ]),
    ];

    for (heading, suite) in groups {
        println!("{heading}");
        for (label, f) in *suite {
            test_suite(label, *f)?;
        }
        println!();
    }

    Ok(())
}
