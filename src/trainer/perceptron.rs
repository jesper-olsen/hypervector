use crate::trainer::{EpochResult, PrototypeModel};
use crate::{Accumulator, HyperVector, nearest};
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

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
    pub fn new(samples: Vec<T>, labels: Vec<L>, rng: R) -> Self {
        assert_eq!(samples.len(), labels.len());

        let mut accumulators: [T::Accumulator; N] = core::array::from_fn(|_| T::Accumulator::new());

        for (hdv, label) in samples.iter().zip(labels.iter()) {
            accumulators[(*label).into()].add(hdv, 1.0);
        }

        let prototypes: [T; N] = core::array::from_fn(|i| accumulators[i].finalize());
        let indices = (0..samples.len()).collect();

        Self {
            accumulators,
            prototypes,
            samples,
            labels,
            indices,
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
        self.indices.shuffle(&mut self.rng); // not needed for batch training...
        let lr = 1.0 / (epoch as f64).sqrt();

        // Parallel: collect misclassifications
        // batch training rather than online where we update after every sample
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
