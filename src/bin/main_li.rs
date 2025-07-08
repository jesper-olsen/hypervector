use clap::Parser;
use hypervector::{
    Accumulator, HyperVector, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV,
    complex_hdv::ComplexHDV, real_hdv::RealHDV,
};
use mersenne_twister_rs::MersenneTwister64;
use rand_core::RngCore;
use std::collections::hash_map::HashMap;
use std::collections::vec_deque::VecDeque;
use std::fs::File;
use std::io::{self, BufRead};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "bipolar", "real", "complex"])]
    mode: String,

    #[arg(long, default_value_t = 1024)]
    /// one of 1024, 10048, 100032
    dim: usize,

    #[arg(long, default_value_t = 3)]
    ngram: usize,
}

const LANGUAGES: [&str; 22] = [
    "af", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "nl",
    "pl", "pt", "ro", "sk", "sl", "sv",
];

pub fn compute_sum_hv<T: HyperVector, R: RngCore>(
    fname: &str,
    n: usize,
    symbols: &mut HashMap<char, T>,
    rng: &mut R,
) -> Result<T, io::Error> {
    let file = File::open(fname)?;
    let reader = io::BufReader::new(file);
    let mut acc = T::Accumulator::default();

    for line in reader.lines().flatten() {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < n {
            continue;
        }
        let mut ngram = T::ident();
        let mut block: VecDeque<char> = VecDeque::with_capacity(n);
        for &c in &chars[..n] {
            let b0 = symbols.entry(c).or_insert(T::random(rng));
            block.push_front(c);
            ngram = ngram.pbind(1, b0, 0);
        }
        for &c in &chars[n..] {
            let forget = block.pop_back().unwrap();
            let forget_sym = symbols.get(&forget).unwrap();
            ngram = ngram.punbind(0, forget_sym, n - 1); // Unbind oldest character (position n-1)
            let new_sym = symbols.entry(c).or_insert(T::random(rng));
            block.push_front(c);
            ngram = ngram.pbind(1, new_sym, 0); // Bind newest character (position 0)
            acc.add(&ngram);
        }
    }
    Ok(acc.finalize())
}

fn train<T: HyperVector, R: RngCore>(
    n: usize,
    rng: &mut R,
) -> Result<(HashMap<char, T>, Vec<(&'static str, T)>), io::Error> {
    let mut symbols: HashMap<char, T> = HashMap::new();
    let mut languages: Vec<(&str, T)> = Vec::new();
    for (i, lxx) in LANGUAGES.iter().enumerate() {
        let fname = format!("DATA/LANG_ID/training_texts/{lxx}.txt");
        println!("{i}/{}: Processing training file {fname}", LANGUAGES.len());
        let v = compute_sum_hv(&fname, n, &mut symbols, rng)?;
        languages.push((lxx, v));
    }
    Ok((symbols, languages))
}

fn test<T: HyperVector, R: RngCore>(
    symbols: &mut HashMap<char, T>,
    languages: &[(&str, T)],
    n: usize,
    rng: &mut R,
) -> Result<(), io::Error> {
    let mut total = 0;
    let mut correct = 0;

    for (i, lxx) in LANGUAGES.iter().enumerate() {
        println!("{i}/{}: Processing {lxx}", LANGUAGES.len());

        let pattern = format!("DATA/LANG_ID/testing_texts/{lxx}_*.txt");
        for fname in glob::glob(&pattern).expect("wrong glob pattern") {
            let fname = fname.unwrap();
            let v = compute_sum_hv(fname.to_str().unwrap(), n, symbols, rng)?;
            let mut min_lang = 0;
            let b = &languages[0].1;
            let mut dmin = T::distance(&v, b);
            for (j, (_lang, b)) in languages.iter().enumerate().skip(1) {
                let d = T::distance(&v, b);
                if d < dmin {
                    dmin = d;
                    min_lang = j;
                }
            }
            if &languages[min_lang].0 == lxx {
                correct += 1;
            }
            total += 1;
        }
        if total > 0 {
            println!("+{} {lxx}: Accuracy: {correct}/{total}={}", i + 1, {
                correct as f64 / total as f64
            })
        }
    }
    Ok(())
}

fn run<T: HyperVector>(n: usize) -> Result<(), io::Error> {
    let mut mt = MersenneTwister64::new(42);
    let (mut symbols, languages) = train::<T, _>(n, &mut mt).expect("Training failed");
    test(&mut symbols, &languages, n, &mut mt)
}

fn main() -> Result<(), io::Error> {
    let args = Args::parse();
    let n = args.ngram;
    println!(
        "Mode: {} N-gram: {} Dim: {}",
        args.mode, args.ngram, args.dim
    );
    match (args.mode.as_str(), args.dim) {
        ("binary", 1024) => run::<BinaryHDV<16>>(n)?,
        ("binary", 10048) => run::<BinaryHDV<157>>(n)?,
        ("binary", 100032) => run::<BinaryHDV<1563>>(n)?,
        ("bipolar", 1024) => run::<BipolarHDV<1024>>(n)?,
        ("bipolar", 10048) => run::<BipolarHDV<10048>>(n)?,
        ("bipolar", 100032) => run::<BipolarHDV<100032>>(n)?,
        // ("real", 1024) => run::<RealHDV<1024>>(n)?,
        // ("real", 10048) => run::<RealHDV<10048>>(n)?,
        // ("real", 100032) => run::<RealHDV<100032>>(n)?,
        // ("complex", 1024) => run::<ComplexHDV<1024>>(n)?,
        // ("complex", 10048) => run::<ComplexHDV<10048>>(n)?,
        // ("complex", 100032) => run::<ComplexHDV<100032>>(n)?,
        _ => {
            eprintln!("Unsupported combination: {:?}", args);
            std::process::exit(1);
        }
    };

    Ok(())
}
