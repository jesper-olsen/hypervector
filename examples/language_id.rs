use clap::Parser;
use hypervector::types::traits::{HyperVector, UnitAccumulator};
use hypervector::types::{
    binary::Binary, bipolar::Bipolar, complex::ComplexHDV, modular::Modular, real::RealHDV,
};
use hypervector::{cleanup, save_hypervectors_to_csv};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use std::collections::hash_map::HashMap;
use std::collections::vec_deque::VecDeque;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "bipolar", "real", "complex", "modular"])]
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

pub fn create_language_profile<T: HyperVector, R: Rng>(
    fname: &Path,
    n: usize,
    symbols: &mut HashMap<char, T>,
    rng: &mut R,
) -> Result<T, io::Error> {
    let file = File::open(fname)?;
    let reader = io::BufReader::new(file);
    let mut acc = T::UnitAccumulator::default();

    for line in reader.lines().map_while(Result::ok) {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < n {
            continue;
        }

        for window in chars.windows(n) {
            // For each window, we recompute the n-gram hypervector from scratch.
            // this works for all kinds of HDVs
            // a potentially faster way for binary and bipolar is to 'unbind' the oldest symbol
            // for real and complex HDVs this doesn't work well because unbind is too noisy
            let mut ngram = T::ident();
            for &c in window.iter() {
                let sym = symbols.entry(c).or_insert_with(|| T::random(rng));
                ngram = ngram.permute(1).bind(sym);
            }
            acc.add(&ngram);
        }
    }
    Ok(acc.finalize())
}

// like create_language_profile - use unbind to update ngrams
// unbind is noisy for real/complex HDVs
pub fn create_language_profile_bind<T: HyperVector, R: Rng>(
    fname: &Path,
    n: usize,
    symbols: &mut HashMap<char, T>,
    rng: &mut R,
) -> Result<T, io::Error> {
    let file = File::open(fname)?;
    let reader = io::BufReader::new(file);
    let mut acc = T::UnitAccumulator::default();

    for line in reader.lines().map_while(Result::ok) {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < n {
            continue;
        }
        let mut ngram = T::ident();
        let mut block: VecDeque<char> = VecDeque::with_capacity(n);
        for &c in &chars[..n] {
            let sym = symbols.entry(c).or_insert_with(|| T::random(rng));
            block.push_front(c);
            ngram = ngram.permute(1).bind(sym);
        }

        acc.add(&ngram);
        for &c in &chars[n..] {
            let forget = block.pop_back().unwrap();
            let forget_sym = symbols.get(&forget).unwrap();

            // Unbind the oldest symbol - it has been permuted (n-1) times
            let to_remove = forget_sym.permute(n - 1);
            ngram = ngram.unbind(&to_remove);

            let new_sym = symbols.entry(c).or_insert_with(|| T::random(rng));
            block.push_front(c);

            // Shift the remaining (n-1) symbols and bind the new one at position 0
            ngram = ngram.permute(1).bind(new_sym);

            acc.add(&ngram);
        }
    }
    Ok(acc.finalize())
}

type SymbolMap<T> = HashMap<char, T>;
type LanguageModel<T> = Vec<(&'static str, T)>;

fn train<T: HyperVector, R: Rng>(
    n: usize,
    rng: &mut R,
) -> Result<(SymbolMap<T>, LanguageModel<T>), io::Error> {
    let mut symbols: HashMap<char, T> = HashMap::new();
    let mut languages: Vec<(&str, T)> = Vec::new();
    for (i, lxx) in LANGUAGES.iter().enumerate() {
        let fname = format!("DATA/LANG_ID/training_texts/{lxx}.txt");
        println!("{i}/{}: Processing training file {fname}", LANGUAGES.len());
        let v = create_language_profile(Path::new(&fname), n, &mut symbols, rng)?;
        languages.push((lxx, v));
    }
    Ok((symbols, languages))
}

fn test<T: HyperVector, R: Rng>(
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
            let fname = fname.map_err(io::Error::other)?;
            let v = create_language_profile(&fname, n, symbols, rng)?;
            if cleanup(&v, languages) == *lxx {
                correct += 1
            }
            total += 1;
        }
        if total > 0 {
            let acc = 100.0 * correct as f64 / total as f64;
            println!("+{} {lxx}: Accuracy: {correct}/{total}={acc:.2}%", i + 1)
        }
    }
    Ok(())
}

fn run<T: HyperVector + Clone>(n: usize) -> Result<(), io::Error> {
    let mut mt = MersenneTwister64::new(42);
    let (mut symbols, languages) = train::<T, _>(n, &mut mt).expect("Training failed");
    let model: Vec<T> = languages.iter().map(|(_label, hdv)| hdv.clone()).collect();
    save_hypervectors_to_csv("RESULTS/model.csv", &model)?;
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
        ("binary", 1024) => run::<Binary<16>>(n)?,
        ("binary", 10048) => run::<Binary<157>>(n)?,
        ("binary", 100032) => run::<Binary<1563>>(n)?,
        ("binary", 200000) => run::<Binary<3125>>(n)?,
        ("bipolar", 1024) => run::<Bipolar<16>>(n)?,
        ("bipolar", 10048) => run::<Bipolar<157>>(n)?,
        ("bipolar", 100032) => run::<Bipolar<1563>>(n)?,
        ("real", 1024) => run::<RealHDV<1024>>(n)?,
        ("real", 10048) => run::<RealHDV<10048>>(n)?,
        ("real", 100032) => run::<RealHDV<100032>>(n)?,
        ("complex", 1024) => run::<ComplexHDV<1024>>(n)?,
        ("complex", 10048) => run::<ComplexHDV<10048>>(n)?,
        ("complex", 100032) => run::<ComplexHDV<100032>>(n)?,
        ("modular", 1024) => run::<Modular<1024>>(n)?,
        ("modular", 10048) => run::<Modular<10048>>(n)?,
        ("modular", 100032) => run::<Modular<100032>>(n)?,
        _ => {
            eprintln!("Unsupported combination: {args:?}");
            std::process::exit(1);
        }
    };

    Ok(())
}
