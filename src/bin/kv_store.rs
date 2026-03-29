// Associative memory test
// 1. Create N key-value pairs for hypervectors of given dimensionality
// 2. Bind keys and values
// 3. Bundle all
// 4. Unbind a key and try to decode the corresponding value.
// Plot the results with plot_kv.py

use hypervector::{
    HyperVector, UnitAccumulator, binary_hdv::BinaryHDV, complex_hdv::ComplexHDV, hdv,
    modular_hdv::ModularHDV, nearest, real_hdv::RealHDV,
};
use mersenne_twister_rs::MersenneTwister64;
use std::io::{self, Write};

hdv!(binary, BinaryHDV1024, 1024);
hdv!(binary, BinaryHDV8192, 8192);
hdv!(binary, BinaryHDV65536, 65536);
hdv!(modular, ModularHDV1024, 1024);
hdv!(modular, ModularHDV8192, 8192);
hdv!(modular, ModularHDV16384, 16384);
hdv!(real, RealHDV1024, 1024);
hdv!(real, RealHDV2048, 2048);
hdv!(complex, ComplexHDV1024, 1024);

const TRIALS: usize = 10;

/// N values to sweep: fine-grained log-ish spacing to capture the capacity cliff
fn n_values() -> Vec<usize> {
    let mut ns = vec![5, 10, 20, 30, 50, 75];
    for n in (100..=1000).step_by(100) {
        ns.push(n);
    }
    ns
}

fn run_trial<T: HyperVector + Clone>(n: usize, seed: u64) -> f64 {
    let mut rng = MersenneTwister64::new(seed);
    let keys: Vec<T> = (0..n).map(|_| T::random(&mut rng)).collect();
    let values: Vec<T> = (0..n).map(|_| T::random(&mut rng)).collect();

    let mut acc = T::UnitAccumulator::new();
    for (k, v) in keys.iter().zip(values.iter()) {
        acc.add(&k.bind(v));
    }
    let bundle = acc.finalize();

    let correct: usize = (0..n)
        .map(|i| {
            let v = bundle.unbind(&keys[i]);
            let (idx, _) = nearest(&v, &values);
            if i == idx { 1 } else { 0 }
        })
        .sum();

    correct as f64 / n as f64
}

/// Returns (mean_accuracy, std_dev) across TRIALS seeds
fn run_averaged<T: HyperVector + Clone>(n: usize) -> (f64, f64) {
    let accs: Vec<f64> = (0..TRIALS as u64)
        .map(|seed| run_trial::<T>(n, seed * 1_000_003))
        .collect();
    let mean = accs.iter().sum::<f64>() / accs.len() as f64;
    let var = accs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / accs.len() as f64;
    (mean * 100.0, var.sqrt() * 100.0)
}

fn sweep<T: HyperVector + Clone>(
    label: &str,
    bits: usize,
    out: &mut impl Write,
) -> Result<(), io::Error> {
    for n in n_values() {
        let (mean, std) = run_averaged::<T>(n);
        writeln!(out, "{label},{bits},{n},{mean:.4},{std:.4}")?;
        //eprintln!("  {label}  n={n:>5}  acc={mean:.1}% ± {std:.1}%");
    }
    Ok(())
}

fn main() -> Result<(), io::Error> {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "type,bits,n,accuracy_mean,accuracy_std")?;

    sweep::<BinaryHDV1024>("Binary dim=1024", 1024, &mut out)?;
    sweep::<BinaryHDV8192>("Binary dim=8192", 8192, &mut out)?;
    sweep::<BinaryHDV65536>("Binary dim=65536", 65536, &mut out)?;
    sweep::<BinaryHDV65536>("Binary dim=131072", 131072, &mut out)?;
    sweep::<ModularHDV1024>("Modular dim=1024", 8192, &mut out)?;
    sweep::<ModularHDV8192>("Modular dim=8192", 65536, &mut out)?;
    sweep::<ModularHDV16384>("Modular dim=16384", 131072, &mut out)?;
    sweep::<RealHDV1024>("Real dim=1024", 65536, &mut out)?;
    //sweep::<RealHDV2048>    ("Real dim=2048",     131072,&mut out)?;
    //sweep::<ComplexHDV1024> ("Complex dim=1024",  131072,&mut out)?;

    Ok(())
}
