use clap::Parser;
use hypervector::hdv;
use hypervector::trainer::lvq::LvqTrainer;
use hypervector::trainer::{
    Classifier, MultiPrototypeModel, PrototypeModel, multi_perceptron::PerceptronMultiTrainer,
    pa::PaTrainer, pa::PaVariant, perceptron::PerceptronTrainer,
};
use hypervector::{Accumulator, HyperVector, trainer::Trainer};
use hypervector::{
    binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV, complex_hdv::ComplexHDV,
    modular_hdv::ModularHDV, real_hdv::RealHDV, save_hypervectors_to_csv,
};

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;

// ls -l UCI\ HAR\ Dataset/test
//
//        352 Nov 30  2012 Inertial Signals
//       7934 Nov 30  2012 subject_test.txt
//   26458166 Nov 30  2012 X_test.txt
//       5894 Nov 30  2012 y_test.txt
//
// ls -l UCI\ HAR\ Dataset/train
//
//        352 Nov 30  2012 Inertial Signals
//      20152 Nov 30  2012 subject_train.txt
//   66006256 Nov 30  2012 X_train.txt
//      14704 Nov 30  2012 y_train.txt

// subject_test.txt and subject_train.txt
// subject id (1..30) one per line, subject on line n refers to feature vector on line n in X_test/X_train/y_test/y_train
//
// y_test.txt and y_train.txt
// activity (1-6), line n refers tofeature v
// activity (1-6) one per line, activity on line n refers to feature vector on line n in X_test/X_train/y_test/y_train

pub const N_FEATURES: usize = 561;
pub type Sample = [f32; N_FEATURES];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Activity {
    Walking = 0,
    WalkingUpstairs = 1,
    WalkingDownstairs = 2,
    Sitting = 3,
    Standing = 4,
    Laying = 5,
}

impl Activity {
    pub fn from_label(n: u8) -> Self {
        match n {
            1 => Activity::Walking,
            2 => Activity::WalkingUpstairs,
            3 => Activity::WalkingDownstairs,
            4 => Activity::Sitting,
            5 => Activity::Standing,
            6 => Activity::Laying,
            _ => panic!("Invalid activity label: {n}"),
        }
    }
}

impl From<Activity> for usize {
    fn from(a: Activity) -> usize {
        a as usize
    }
}

pub struct HarDataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
    pub train_labels: Vec<Activity>,
    pub test_labels: Vec<Activity>,
    pub train_subjects: Vec<u8>,
    pub test_subjects: Vec<u8>,
}

impl HarDataset {
    pub fn load(dir: &str) -> io::Result<Self> {
        let base = Path::new(dir);
        Ok(Self {
            train: load_features(&base.join("train/X_train.txt"))?,
            test: load_features(&base.join("test/X_test.txt"))?,
            train_labels: load_labels(&base.join("train/y_train.txt"))?,
            test_labels: load_labels(&base.join("test/y_test.txt"))?,
            train_subjects: load_subjects(&base.join("train/subject_train.txt"))?,
            test_subjects: load_subjects(&base.join("test/subject_test.txt"))?,
        })
    }
}

fn load_features(path: &Path) -> io::Result<Vec<Sample>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line?;
            let mut sample = [0.0f32; N_FEATURES];
            let mut i = 0;
            for tok in line.split_whitespace() {
                sample[i] = f32::from_str(tok)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                i += 1;
            }
            if i != N_FEATURES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected {N_FEATURES} features, got {i}"),
                ));
            }
            Ok(sample)
        })
        .collect()
}

fn load_labels(path: &Path) -> io::Result<Vec<Activity>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            let n: u8 = line?
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Activity::from_label(n))
        })
        .collect()
}

fn load_subjects(path: &Path) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            line?
                .trim()
                .parse::<u8>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect()
}

pub struct HarEncoder<T: HyperVector> {
    base_vectors: Vec<T>, // one per feature, length 561
}

impl<T: HyperVector> HarEncoder<T> {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let base_vectors = (0..561).map(|_| T::random(rng)).collect();
        Self { base_vectors }
    }

    pub fn encode(&self, features: &[f32]) -> T {
        let mut acc = T::Accumulator::new();
        for (base, &val) in self.base_vectors.iter().zip(features.iter()) {
            // val in [-1, 1]: use as weight directly
            acc.add(base, val as f64);
        }
        acc.finalize()
    }
}
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "bipolar", "real", "complex", "modular"])]
    mode: String,

    #[arg(long, default_value_t = 1024)]
    /// one of 1024, 10048, 100032
    dim: usize,

    #[arg(long, default_value = "perceptron", value_parser=["perceptron", "pa", "pai", "paii", "multi", "lvq"])]
    trainer: String,

    #[arg(long, default_value_t = 1)]
    /// number of prototypes per class
    prototypes: usize,

    #[arg(long, default_value_t = 0.25)]
    /// lvq window
    window: f32,

    #[arg(long, default_value_t = 1000)]
    epochs: usize,
}

hdv!(binary, BinaryHDV1024, 1024);
hdv!(binary, BinaryHDV8192, 8192);
hdv!(binary, BinaryHDV16384, 16384);
hdv!(modular, ModularHDV1024, 1024);
hdv!(modular, ModularHDV8192, 8192);
hdv!(modular, ModularHDV16384, 16384);
hdv!(real, RealHDV1024, 1024);
hdv!(real, RealHDV2048, 2048);
hdv!(complex, ComplexHDV1024, 1024);

fn main() -> Result<(), io::Error> {
    let args = Args::parse();
    println!("Mode: {} ", args.mode);
    //match (args.mode.as_str(), args.dim) {
    //    ("binary", 1024) => run::<BinaryHDV<16>>(n)?,
    //    ("binary", 100032) => run::<BinaryHDV<1563>>(n)?,
    //    ("binary", 200000) => run::<BinaryHDV<3125>>(n)?,
    //    _ => {
    //        eprintln!("Unsupported combination: {args:?}");
    //        std::process::exit(1);
    //    }
    //};

    let har = HarDataset::load("UCI HAR Dataset")?;
    println!(
        "#train samples: {} #labels: {}",
        har.train.len(),
        har.train_labels.len()
    );
    println!(
        "#test samples: {} #labels: {}",
        har.test.len(),
        har.test_labels.len()
    );
    println!("train subjects: {}", har.train_subjects.len());
    println!("test subjects: {}", har.test_subjects.len());

    let mut rng = MersenneTwister64::default();

    //run::<BinaryHDV1024>(&har, &mut rng);
    //run::<BinaryHDV8192>(&har, &mut rng);
    run::<BinaryHDV16384>(&har, &mut rng, &args);

    Ok(())
}

fn train<T, Tr>(mut trainer: Tr, epochs: usize) -> Tr::Model
where
    T: HyperVector,
    Tr: Trainer<T>,
{
    for epoch in 1..=epochs {
        let r = trainer.step(epoch);
        print!(
            "Epoch {epoch}: {}/{}={:.2}%\r",
            r.correct,
            r.total(),
            r.accuracy() * 100.0
        );
        std::io::stdout().flush().unwrap();
        if r.errors == 0 {
            break;
        }
    }
    println!();
    trainer.into_model()
}

fn run<T: HyperVector + Sync + Send>(data: &HarDataset, rng: &mut impl Rng, args: &Args) {
    let encoder = HarEncoder::<T>::new(rng);

    let train_hvs: Vec<T> = data.train.par_iter().map(|s| encoder.encode(s)).collect();
    let test_hvs: Vec<T> = data.test.par_iter().map(|s| encoder.encode(s)).collect();

    const NUM_CLASSES: usize = 6;
    let k = args.prototypes;
    let epochs = args.epochs;

    match args.trainer.as_str() {
        "perceptron" => report(
            train(
                PerceptronTrainer::<T, Activity, _, NUM_CLASSES>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    rng,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        "pa" => report(
            train(
                PaTrainer::<T, Activity, _, NUM_CLASSES>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    PaVariant::Pa,
                    rng,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        "pai" => report(
            train(
                PaTrainer::<T, Activity, _, NUM_CLASSES>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    PaVariant::PaI { c: 0.1 },
                    rng,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        "paii" => report(
            train(
                PaTrainer::<T, Activity, _, NUM_CLASSES>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    PaVariant::PaII { c: 1.0 },
                    rng,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        "multi" => report(
            train(
                PerceptronMultiTrainer::<T, _>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    NUM_CLASSES,
                    k,
                    rng,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        "lvq" => report(
            train(
                LvqTrainer::<T, _>::new(
                    train_hvs,
                    data.train_labels.clone(),
                    NUM_CLASSES,
                    k,
                    rng,
                    args.window,
                ),
                epochs,
            ),
            &test_hvs,
            &data.test_labels,
        ),
        _ => eprintln!("Unknown trainer: {}", args.trainer),
    }
}

fn report<T, C>(model: C, test_hvs: &[T], labels: &[Activity])
where
    T: HyperVector + Send + Sync,
    C: Classifier<T> + Sync,
{
    let (correct, errors, acc) = model.accuracy(test_hvs, labels);
    println!("Test Accuracy: {correct}/{}={:.2}%", correct + errors, acc);
}
