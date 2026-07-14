use clap::Parser;
use hypervector::encoding::ScalarEncoder;
use hypervector::types::binary::Binary;
use hypervector::types::traits::{Accumulator, HyperVector};
use hypervector::{
    hdv,
    trainer::{argmin, ensemble_fusion, perceptron::PerceptronTrainer},
};
use mnist::{self, Image, Mnist, error::MnistError};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;

/// A demo application to showcase mnist handwritten digit classification
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the directory containing the MNIST dataset files.
    #[arg(short, long, default_value = "MNIST")]
    data_dir: PathBuf,

    /// Number of individual classifiers to train
    #[arg(short, long, default_value_t = 5)]
    ensemble_size: usize,

    /// Augment training images by jittering
    #[arg(long, default_value_t = false)]
    augment: bool,
}

const FEATURE_PIXEL_BAG: u8 = 1;
const FEATURE_HORIZONTAL: u8 = 2;
const FEATURE_VERTICAL: u8 = 4;
const FEATURE_DIAGONAL1: u8 = 8;
const FEATURE_DIAGONAL2: u8 = 16;
const FEATURE_THINNED: u8 = 32;
const FEATURE_EDGES: u8 =
    FEATURE_HORIZONTAL | FEATURE_VERTICAL | FEATURE_DIAGONAL1 | FEATURE_DIAGONAL2;

/// Feature extraction for MNIST digit images using hyperdimensional computing.
///
/// Encodes images into high-dimensional binary vectors by combining:
/// - Pixel bag-of-words (position × intensity)
/// - Edge features (4 orientations via Sobel operators)
pub struct MnistEncoder<T: HyperVector> {
    positions: Vec<T>,             // one for each of 784 pixels (28x28)
    intensities: ScalarEncoder<T>, // 256 gray levels (0-255)
    feature_horizontal_edge: T,
    feature_vertical_edge: T,
    feature_diag_tl_br: T,
    feature_diag_tr_bl: T,
    feature_thinned: T,
    features: u8,
}

impl<T: HyperVector> MnistEncoder<T> {
    pub fn new(mut rng: &mut impl Rng) -> Self {
        let positions = (0..784).map(|_| T::random(&mut rng)).collect();

        let intensities = ScalarEncoder::<T>::new(0.0, 255.0, 256, rng);

        MnistEncoder {
            positions,
            intensities,
            feature_horizontal_edge: T::random(&mut rng),
            feature_vertical_edge: T::random(&mut rng),
            feature_diag_tl_br: T::random(&mut rng),
            feature_diag_tr_bl: T::random(&mut rng),
            feature_thinned: T::random(&mut rng),
            features: 0,
        }
    }

    pub fn with_feature_pixel_bag(mut self) -> Self {
        self.features |= FEATURE_PIXEL_BAG;
        self
    }

    pub fn with_feature_horizontal(mut self) -> Self {
        self.features |= FEATURE_HORIZONTAL;
        self
    }

    pub fn with_feature_vertical(mut self) -> Self {
        self.features |= FEATURE_VERTICAL;
        self
    }

    pub fn with_feature_diagonal1(mut self) -> Self {
        self.features |= FEATURE_DIAGONAL1;
        self
    }

    pub fn with_feature_diagonal2(mut self) -> Self {
        self.features |= FEATURE_DIAGONAL2;
        self
    }

    pub fn with_feature_edges(mut self) -> Self {
        self.features |= FEATURE_EDGES;
        self
    }

    pub fn with_feature_thinned(mut self) -> Self {
        self.features |= FEATURE_THINNED;
        self
    }

    fn encode(&self, image: &Image) -> T {
        let mut accumulator = T::Accumulator::new();
        const EDGE_THRESHOLD: i16 = 100; // tunable
        let pixels = image.as_u8_array();
        let width = image.width();
        let height = image.height();

        let features = if self.features == 0 {
            FEATURE_PIXEL_BAG | FEATURE_EDGES
        } else {
            self.features
        };

        if features & FEATURE_PIXEL_BAG != 0 {
            const THRESHOLD: u8 = 0;

            for (i, &intensity) in pixels.iter().enumerate() {
                if intensity > THRESHOLD {
                    let intensity_hdv = self.intensities.encode(intensity as f32);
                    let pixel_hdv = self.positions[i].bind(intensity_hdv);
                    let weight = intensity as f64 / 255.0;
                    accumulator.add(&pixel_hdv, weight);
                }
            }
        }

        // 3x3 Sobel Operator
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                // Get intensities of all 9 pixels in the 3x3 grid centered at (x, y)
                let p00 = pixels[(y - 1) * width + (x - 1)] as i16;
                let p01 = pixels[(y - 1) * width + x] as i16;
                let p02 = pixels[(y - 1) * width + (x + 1)] as i16;
                let p10 = pixels[y * width + (x - 1)] as i16;
                // p11 is the center pixel
                let p12 = pixels[y * width + (x + 1)] as i16;
                let p20 = pixels[(y + 1) * width + (x - 1)] as i16;
                let p21 = pixels[(y + 1) * width + x] as i16;
                let p22 = pixels[(y + 1) * width + (x + 1)] as i16;
                let center_idx = y * width + x;

                const EDGE_WEIGHT: f64 = 4.0;
                if features & FEATURE_HORIZONTAL != 0 {
                    // Horizontal Gradient (Sobel Gx): (right side) - (left side)
                    let diff = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
                    //// --- Check for Horizontal Edge --- (comparing tl and tr)
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_horizontal_edge);
                        // note - could use use diff instead to discriminate between white->black, black->white,
                        // but in practise seems worse
                        let magnitude = (diff.abs() as f64) / 1020.0; // Normalize
                        accumulator.add(&feature_hdv, EDGE_WEIGHT * magnitude);
                    }
                }

                if features & FEATURE_VERTICAL != 0 {
                    // Vertical Gradient (Sobel Gy): (bottom side) - (top side)
                    let diff = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                    //// --- Check for Vertical Edge --- (comparing tl and bl)
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_vertical_edge);
                        let magnitude = (diff.abs() as f64) / 1020.0; // Normalize
                        accumulator.add(&feature_hdv, EDGE_WEIGHT * magnitude);
                    }
                }
                if features & FEATURE_DIAGONAL1 != 0 {
                    // --- Check for Diagonal Edge (/) --- (comparing tr and bl)
                    // Kernel: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
                    let diff = (p01 + 2 * p02 + p12) - (p10 + 2 * p20 + p21);
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tl_br);
                        let magnitude = (diff.abs() as f64) / 1020.0; // Normalize
                        accumulator.add(&feature_hdv, EDGE_WEIGHT * magnitude);
                    }
                }

                if features & FEATURE_DIAGONAL2 != 0 {
                    //// --- Check for Diagonal Edge (\) --- (comparing tl and br)
                    // Kernel: [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
                    let diff = (2 * p00 + p01 + p10) - (p12 + p21 + 2 * p22);
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tr_bl);
                        let magnitude = (diff.abs() as f64) / 1020.0; // Normalize
                        accumulator.add(&feature_hdv, EDGE_WEIGHT * magnitude);
                    }
                }
            }
        }

        if features & FEATURE_THINNED != 0 {
            // Stack-allocated buffer for the 28x28 MNIST image
            let mut bits = [false; 784];

            // Binarize - adjustable threshold
            for (i, &p) in pixels.iter().enumerate() {
                bits[i] = p > 127;
            }

            zhang_suen::thinning(&mut bits, width);

            // Bind the positions of the remaining skeleton pixels
            for (i, &is_skeleton) in bits.iter().enumerate() {
                if is_skeleton {
                    let feature_hdv = self.positions[i].bind(&self.feature_thinned);
                    // Skeleton pixels are binary, so we use a full weight of 1.0
                    accumulator.add(&feature_hdv, 1.0);
                }
            }
        }

        accumulator.finalize()
    }
}

/// Computes the majority vote for a set of predictions.
///
/// `predictions` is a slice of prediction arrays, where each array comes from a single model.
/// `num_classes` defines the size of the voting ballot.
pub fn majority_vote(predictions: &[Vec<u8>], num_classes: usize) -> Vec<u8> {
    if predictions.is_empty() {
        return Vec::new();
    }

    let n_test = predictions[0].len();

    (0..n_test)
        .into_par_iter()
        .map(|i| {
            let mut counts = vec![0u32; num_classes];
            for r in predictions {
                counts[r[i] as usize] += 1;
            }
            let mut best_label = 0;
            let mut best_count = 0;
            for (label, count) in counts.into_iter().enumerate() {
                if count >= best_count {
                    best_label = label;
                    best_count = count;
                }
            }
            best_label as u8
        })
        .collect()
}

// -- Confusion Matrix ------------------------------------------------------

/// Row = true label, column = predicted label.
fn confusion_matrix(predictions: &[u8], labels: &[u8]) -> [[u32; 10]; 10] {
    let mut m = [[0u32; 10]; 10];
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        m[l as usize][p as usize] += 1;
    }
    m
}

fn print_confusion_matrix(m: &[[u32; 10]; 10]) {
    print!("true\\pred");
    for j in 0..10 {
        print!("{j:6}");
    }
    println!();
    for (i, row) in m.iter().enumerate() {
        print!("{i:8} ");
        for &count in row {
            print!("{count:6}");
        }
        println!();
    }
}

fn main() -> Result<(), MnistError> {
    let args = Args::parse();

    //const TOTAL_BITS: usize = 6400;
    const TOTAL_BITS: usize = 12800;
    hdv!(binary, HDV, TOTAL_BITS);
    let data = if args.augment {
        let max_shift = 1;
        Mnist::load_with_shift_augmentation(args.data_dir, max_shift)?
    } else {
        Mnist::load(args.data_dir)?
    };
    println!("Loaded {} training labels", data.train_labels.len());

    const N: usize = mnist::NUM_LABELS;
    let seed = 42;
    let mut ensemble_hard_results: Vec<Vec<u8>> = Vec::with_capacity(args.ensemble_size);
    let mut ensemble_score_results: Vec<Vec<[f32; N]>> = Vec::with_capacity(args.ensemble_size);
    let mut test_accuracies = Vec::with_capacity(args.ensemble_size);

    for mn in 1..=args.ensemble_size {
        println!("Training model {mn}/{}", args.ensemble_size);
        let mut rng = StdRng::seed_from_u64(seed + mn as u64);
        let n_epochs = 2000;

        let encoder = MnistEncoder::<HDV>::new(&mut rng)
            .with_feature_pixel_bag()
            .with_feature_edges();

        println!("Encoding images (Dim {})...", HDV::DIM);
        let train_hvs: Vec<HDV> = data
            .train_images
            .par_iter()
            .map(|im| encoder.encode(im))
            .collect();
        let mut trainer =
            PerceptronTrainer::<HDV, u8, _, N>::new(&train_hvs, &data.train_labels, None, rng);

        for epoch in 1..=n_epochs {
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
        let model = trainer.into_model();

        let scores: Vec<[f32; N]> = data
            .test_images
            .par_iter()
            .map(|im| model.scores(&encoder.encode(im)))
            .collect();

        let hard_results: Vec<u8> = scores.iter().map(|s| argmin(s) as u8).collect();

        let correct: usize = hard_results
            .par_iter()
            .zip(data.test_labels.par_iter())
            .filter(|&(predicted, label)| *predicted == *label)
            .count();
        let total = data.test_images.len();
        let acc = 100.0 * correct as f64 / total as f64;

        test_accuracies.push(acc);
        ensemble_hard_results.push(hard_results);
        ensemble_score_results.push(scores);
        println!("Test Accuracy: {correct:5}/{total} = {acc:.2}%\n");
    }

    let min = test_accuracies
        .iter()
        .fold(100.0, |acc, x| if *x < acc { *x } else { acc });
    let max = test_accuracies
        .iter()
        .fold(0.0, |acc, x| if *x > acc { *x } else { acc });
    println!("Accuracy range: {min:.1}% - {max:.1}%");
    let n_test = data.test_labels.len();

    // Hard voting baseline (majority of per-model argmin labels)
    let hard_vote_predictions = majority_vote(&ensemble_hard_results, N);
    let hard_vote_correct = hard_vote_predictions
        .par_iter()
        .zip(data.test_labels.par_iter())
        .filter(|&(p, l)| p == l)
        .count();
    let hard_vote_acc = 100.0 * hard_vote_correct as f64 / n_test as f64;
    println!(
        "\nHard-vote Ensemble Accuracy:   {hard_vote_correct:5}/{n_test} = {hard_vote_acc:.2}%"
    );

    // Sum-rule score fusion
    let fusion_predictions: Vec<u8> = ensemble_fusion(&ensemble_score_results)
        .into_iter()
        .map(|l| l as u8)
        .collect();
    let fusion_correct = fusion_predictions
        .par_iter()
        .zip(data.test_labels.par_iter())
        .filter(|&(p, l)| p == l)
        .count();
    let fusion_acc = 100.0 * fusion_correct as f64 / n_test as f64;
    println!("Score-fusion Ensemble Accuracy: {fusion_correct:5}/{n_test} = {fusion_acc:.2}%");

    println!("\nScore-fusion Confusion Matrix:");
    let cm = confusion_matrix(&fusion_predictions, &data.test_labels);
    print_confusion_matrix(&cm);

    Ok(())
}
