use crate::{Accumulator, HyperVector, nearest};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// Result of a single training epoch.
#[derive(Debug, Clone, Copy)]
pub struct EpochResult {
    pub epoch: usize,
    pub correct: usize,
    pub errors: usize,
}

impl EpochResult {
    pub fn total(&self) -> usize {
        self.correct + self.errors
    }

    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.total() as f64
    }
}

/// A trained set of prototype hypervectors, one per class.
/// Returned by the trainer; used for inference.
pub struct PrototypeModel<T: HyperVector, const N: usize> {
    pub prototypes: [T; N],
}

impl<T: HyperVector, const N: usize> PrototypeModel<T, N> {
    /// Predict the class index for an already-encoded hypervector.
    pub fn predict(&self, h: &T) -> usize {
        let (idx, _) = nearest(h, &self.prototypes);
        idx
    }

    /// Measure accuracy over a set of pre-encoded samples.
    pub fn accuracy<L>(&self, samples: &[(T, L)]) -> f64
    where
        L: Into<usize> + Copy,
    {
        let correct = samples
            .iter()
            .filter(|(h, label)| self.predict(h) == (*label).into())
            .count();
        correct as f64 / samples.len() as f64
    }
}

/// Perceptron trainer for HDV prototype classifiers.
///
/// # Type parameters
/// - `T`: HyperVector type
/// - `R`: RNG (used for per-epoch shuffling)
/// - `L`: Label type — must be convertible to `usize` as a class index
/// - `N`: Number of classes (const generic)
pub struct PerceptronTrainer<T, L, R, const N: usize>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    accumulators: [T::Accumulator; N],
    prototypes: [T; N],
    samples: Vec<T>,
    labels: Vec<L>,
    indices: Vec<usize>,
    rng: R,
}

impl<T, L, R, const N: usize> PerceptronTrainer<T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    /// Create a new trainer from an iterator of `(encoded_hdv, label)` pairs.
    ///
    /// The iterator is consumed and the samples are stored internally for
    /// repeated shuffled passes during training.
    pub fn new(hvs: Vec<T>, labels: Vec<L>, rng: R) -> Self {
        let n = hvs.len();
        assert_eq!(n, labels.len());

        let mut accumulators: [T::Accumulator; N] = core::array::from_fn(|_| T::Accumulator::new());

        for (hdv, label) in hvs.iter().zip(labels.iter()) {
            accumulators[(*label).into()].add(hdv, 1.0);
        }

        let prototypes: [T; N] = core::array::from_fn(|i| accumulators[i].finalize());

        Self {
            accumulators,
            prototypes,
            samples: hvs, // owned directly, no tuple packing
            labels,
            indices: (0..n).collect(),
            rng,
        }
    }

    /// Run a single training epoch (perceptron update rule).
    ///
    /// Shuffles the sample order, finds all misclassifications in parallel,
    /// then applies weight updates sequentially.
    ///
    /// `epoch` is 1-based and used to compute the learning rate `1/sqrt(epoch)`.
    pub fn step(&mut self, epoch: usize) -> EpochResult {
        self.indices.shuffle(&mut self.rng);
        let lr = 1.0 / (epoch as f64).sqrt();

        // Parallel: collect misclassifications
        let errors: Vec<(usize, usize, usize)> = self
            .indices
            .par_iter()
            .filter_map(|&idx| {
                let hdv = &self.samples[idx];
                let true_class = self.labels[idx].into();
                let (predicted, _) = nearest(hdv, &self.prototypes);
                if predicted != true_class {
                    Some((idx, true_class, predicted))
                } else {
                    None
                }
            })
            .collect();

        let error_count = errors.len();

        // Sequential: apply updates (accumulators not thread-safe)
        for (idx, true_class, predicted) in errors {
            let hdv = &self.samples[idx];
            self.accumulators[true_class].add(hdv, lr);
            self.accumulators[predicted].add(hdv, -lr);
        }

        self.prototypes = core::array::from_fn(|i| self.accumulators[i].finalize());

        EpochResult {
            epoch,
            correct: self.indices.len() - error_count,
            errors: error_count,
        }
    }

    /// Run up to `max_epochs` training steps, stopping early if zero errors.
    ///
    /// Returns the results of each epoch and the trained model.
    pub fn fit(mut self, max_epochs: usize) -> (PrototypeModel<T, N>, Vec<EpochResult>) {
        let mut history = Vec::with_capacity(max_epochs);
        for epoch in 1..=max_epochs {
            let result = self.step(epoch);
            history.push(result);
            if result.errors == 0 {
                break;
            }
        }
        (
            PrototypeModel {
                prototypes: self.prototypes,
            },
            history,
        )
    }

    /// Borrow the current prototype hypervectors without consuming the trainer.
    /// Useful for evaluating mid-training when driving the loop manually.
    pub fn prototypes(&self) -> &[T; N] {
        &self.prototypes
    }

    /// Consume the trainer and return the final trained model.
    pub fn into_model(self) -> PrototypeModel<T, N> {
        PrototypeModel {
            prototypes: self.prototypes,
        }
    }
}

/// Margin-based Perceptron trainer.
pub struct MarginTrainer<T, L, R, const N: usize>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    accumulators: [T::Accumulator; N],
    prototypes: [T; N],
    samples: Vec<T>,
    labels: Vec<L>,
    indices: Vec<usize>,
    rng: R,
    pub margin: f64, // Threshold for confidence
}

impl<T, L, R, const N: usize> MarginTrainer<T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    pub fn new(hvs: Vec<T>, labels: Vec<L>, rng: R, margin: f64) -> Self {
        let n = hvs.len();
        let mut accumulators: [T::Accumulator; N] = core::array::from_fn(|_| T::Accumulator::new());

        for (hdv, label) in hvs.iter().zip(labels.iter()) {
            accumulators[(*label).into()].add(hdv, 1.0);
        }

        let prototypes: [T; N] = core::array::from_fn(|i| accumulators[i].finalize());

        Self {
            accumulators,
            prototypes,
            samples: hvs,
            labels,
            indices: (0..n).collect(),
            rng,
            margin,
        }
    }

    pub fn step(&mut self, epoch: usize) -> EpochResult {
        self.indices.shuffle(&mut self.rng);
        let lr = 1.0 / (epoch as f64).sqrt();

        // Destructure to local references to avoid capturing `&self` in par_iter
        let prototypes = &self.prototypes;
        let samples = &self.samples;
        let labels = &self.labels;
        let margin = self.margin;

        let violations: Vec<(usize, usize, usize)> = self
            .indices
            .par_iter()
            .filter_map(|&idx| {
                let hdv = &samples[idx];
                let true_class = labels[idx].into();

                // Use the standalone helper function
                let (best_idx, best_sim, second_idx, second_sim) = find_top_two(hdv, prototypes);

                if best_idx != true_class {
                    Some((idx, true_class, best_idx))
                } else if (best_sim - second_sim) < margin {
                    // Update even if correct, because the gap is too small
                    Some((idx, true_class, second_idx))
                } else {
                    None
                }
            })
            .collect();

        let error_count = violations.len();

        for (idx, true_class, competitor) in violations {
            let hdv = &self.samples[idx];
            self.accumulators[true_class].add(hdv, lr);
            self.accumulators[competitor].add(hdv, -lr);
        }

        self.prototypes = core::array::from_fn(|i| self.accumulators[i].finalize());

        EpochResult {
            epoch,
            correct: self.indices.len() - error_count,
            errors: error_count,
        }
    }

    /// Consume the trainer and return the final trained model.
    pub fn into_model(self) -> PrototypeModel<T, N> {
        PrototypeModel {
            prototypes: self.prototypes,
        }
    }
}

/// Standalone helper to find the top two closest prototypes.
/// This is outside the impl block so it doesn't need to capture `self`.
fn find_top_two<T: HyperVector, const N: usize>(
    h: &T,
    prototypes: &[T; N],
) -> (usize, f64, usize, f64) {
    let mut best = (0, f32::MAX);
    let mut second = (0, f32::MAX);

    for (i, proto) in prototypes.iter().enumerate() {
        let sim = h.distance(proto);
        if sim < best.1 {
            second = best;
            best = (i, sim);
        } else if sim < second.1 {
            second = (i, sim);
        }
    }
    (best.0, best.1.into(), second.0, second.1.into())
}
