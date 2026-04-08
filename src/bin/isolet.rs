use clap::{Parser, ValueEnum};
use hypervector::encoding::BundleEncoder;
use hypervector::trainer::isolet_dataset::{Dataset, Label, N_FEATURES, NUM_CLASSES};
use hypervector::trainer::{
    Classifier, Trainer, ensemble_accuracy, lvq::LvqTrainer,
    multi_perceptron::PerceptronMultiTrainer, pa::PaTrainer, pa::PaVariant,
    perceptron::PerceptronTrainer,
};
use hypervector::{HyperVector, hdv};
use hypervector::{
    binary_hdv::BinaryHDV, complex_hdv::ComplexHDV, modular_hdv::ModularHDV, real_hdv::RealHDV,
};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::io;
use std::io::Write;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TrainerKind {
    Perceptron,
    Pa,
    Pai,
    Paii,
    Multi,
    Lvq,
}

impl fmt::Display for TrainerKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainerKind::Perceptron => write!(f, "perceptron"),
            TrainerKind::Pa => write!(f, "pa"),
            TrainerKind::Pai => write!(f, "pai"),
            TrainerKind::Paii => write!(f, "paii"),
            TrainerKind::Multi => write!(f, "multi"),
            TrainerKind::Lvq => write!(f, "lvq"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "bipolar", "real", "complex", "modular"])]
    mode: String,

    #[arg(long, default_value_t = 8192, value_parser = valid_dim)]
    /// One of 1024, 2048, 4096, 8192, 16384
    dim: usize,

    #[arg(long, default_value_t = TrainerKind::Perceptron)]
    trainer: TrainerKind,

    #[arg(long, default_value_t = 1)]
    /// number of prototypes per class
    prototypes: usize,

    #[arg(long, default_value_t = 0.25)]
    /// lvq window
    window: f32,

    #[arg(long, default_value_t = 1000)]
    epochs: usize,

    #[arg(long, default_value_t = 9)]
    ensemble_size: usize,
}

fn valid_dim(s: &str) -> Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("{s} is not a number"))?;
    match n {
        1024 | 2048 | 4096 | 8192 | 16384 => Ok(n),
        _ => Err(format!(
            "{n} is not a supported dimension (1024, 2048, 8192, 16384)"
        )),
    }
}

hdv!(binary, BinaryHDV1024, 1024);
hdv!(binary, BinaryHDV2048, 2048);
hdv!(binary, BinaryHDV4096, 4096);
hdv!(binary, BinaryHDV8192, 8192);
hdv!(binary, BinaryHDV16384, 16384);
hdv!(modular, ModularHDV1024, 1024);
hdv!(modular, ModularHDV2048, 2048);
hdv!(modular, ModularHDV4096, 4096);
hdv!(modular, ModularHDV8192, 8192);
hdv!(modular, ModularHDV16384, 16384);
hdv!(real, RealHDV1024, 1024);
hdv!(real, RealHDV2048, 2048);
hdv!(complex, ComplexHDV1024, 1024);

fn train<T, Tr>(mut trainer: Tr, epochs: usize) -> Tr::Model
where
    T: HyperVector,
    Tr: Trainer<T>,
{
    for epoch in 1..=epochs {
        let r = trainer.step(epoch);
        print!(
            "Epoch {epoch}: Training Accuracy {}/{}={:.2}%\r",
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

fn run<T: HyperVector + Sync + Send>(
    data: &Dataset,
    rng: &mut impl Rng,
    args: &Args,
) -> Vec<usize> {
    let encoder = BundleEncoder::<T, N_FEATURES>::new(rng);
    let train_hvs: Vec<T> = data.train.par_iter().map(|s| encoder.encode(s)).collect();
    let test_hvs: Vec<T> = data.test.par_iter().map(|s| encoder.encode(s)).collect();

    let k = args.prototypes;
    let epochs = args.epochs;

    match args.trainer {
        TrainerKind::Perceptron => {
            let trainer = PerceptronTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                &mut *rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Pa => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                PaVariant::Pa,
                &mut *rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Pai => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                PaVariant::PaI { c: 0.1 },
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Paii => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                PaVariant::PaII { c: 1.0 },
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Multi => {
            let trainer = PerceptronMultiTrainer::<T, _>::new(
                &train_hvs,
                &data.train_labels,
                NUM_CLASSES,
                k,
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Lvq => {
            let trainer = LvqTrainer::<T, _>::new(
                &train_hvs,
                &data.train_labels,
                NUM_CLASSES,
                k,
                rng,
                args.window,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
    }
}

fn stats(v: &[f64]) -> Option<(f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }

    let mut min = v[0];
    let mut max = v[0];
    let mut sum = v[0];

    for &x in &v[1..] {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x;
    }

    let avg = sum / v.len() as f64;
    Some((100.0 * min, 100.0 * max, 100.0 * avg))
}

fn main() -> Result<(), io::Error> {
    let args = Args::parse();
    let ensemble_size = args.ensemble_size;
    let har = Dataset::load("isolet")?;
    let mut rng = MersenneTwister64::default();

    let mut all_predictions = Vec::with_capacity(ensemble_size);
    let mut accs = Vec::with_capacity(ensemble_size);
    for i in 1..=ensemble_size {
        let preds = match (args.mode.as_str(), args.dim) {
            ("binary", 1024) => run::<BinaryHDV1024>(&har, &mut rng, &args),
            ("binary", 2048) => run::<BinaryHDV2048>(&har, &mut rng, &args),
            ("binary", 4096) => run::<BinaryHDV4096>(&har, &mut rng, &args),
            ("binary", 8192) => run::<BinaryHDV8192>(&har, &mut rng, &args),
            ("binary", 16384) => run::<BinaryHDV16384>(&har, &mut rng, &args),
            ("modular", 1024) => run::<ModularHDV1024>(&har, &mut rng, &args),
            ("modular", 2048) => run::<ModularHDV2048>(&har, &mut rng, &args),
            ("modular", 4096) => run::<ModularHDV4096>(&har, &mut rng, &args),
            ("modular", 8192) => run::<ModularHDV8192>(&har, &mut rng, &args),
            ("modular", 16384) => run::<ModularHDV16384>(&har, &mut rng, &args),
            ("real", 1024) => run::<RealHDV1024>(&har, &mut rng, &args),
            ("real", 2048) => run::<RealHDV2048>(&har, &mut rng, &args),
            ("complex", 1024) => run::<ComplexHDV1024>(&har, &mut rng, &args),
            _ => {
                eprintln!(
                    "Unsupported combination: mode={} dim={}",
                    args.mode, args.dim
                );
                return Ok(());
            }
        };
        let (correct, errors, acc) =
            ensemble_accuracy(std::slice::from_ref(&preds), &har.test_labels, NUM_CLASSES);

        println!(
            "Model {i}/{ensemble_size} - test: {:.2}%  ({correct}/{})",
            acc * 100.0,
            correct + errors,
        );
        all_predictions.push(preds);
        accs.push(acc);
        if i > 2 {
            let (correct, errors, acc) =
                ensemble_accuracy(&all_predictions, &har.test_labels, NUM_CLASSES);
            println!(
                "Ensemble of {i} - test {:.2}%  ({correct}/{})",
                acc * 100.0,
                correct + errors,
            );
        }
        println!();
    }

    if let Some((min, max, avg)) = stats(&accs) {
        println!("Model accuracies - avg {avg:.2}%, min {min:.2}%, max {max:.2}")
    }

    Ok(())
}
