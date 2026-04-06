use crate::trainer::{EpochResult, PrototypeModel, Trainer};
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
pub struct PerceptronTrainer<'a, T, L, R, const N: usize>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    accumulators: [T::Accumulator; N],
    prototypes: [T; N],
    samples: &'a [T],
    labels: &'a [L],
    indices: Vec<usize>,
    rng: R,
}

impl<'a, T, L, R, const N: usize> PerceptronTrainer<'a, T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    pub fn new(samples: &'a [T], labels: &'a [L], rng: R) -> Self {
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

    pub fn step(&mut self, epoch: usize) -> EpochResult {
        self.indices.shuffle(&mut self.rng);
        let lr = 1.0 / (epoch as f64).sqrt();

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

    pub fn fit(mut self, max_epochs: usize) -> (PrototypeModel<T, N>, Vec<EpochResult>) {
        let mut history = Vec::with_capacity(max_epochs);
        for epoch in 1..=max_epochs {
            let result = self.step(epoch);
            history.push(result);
            if result.errors == 0 {
                break;
            }
        }
        (self.into_model(), history)
    }

    pub fn into_model(self) -> PrototypeModel<T, N> {
        PrototypeModel {
            prototypes: self.prototypes,
        }
    }
}

impl<'a, T, L, R, const N: usize> Trainer<T> for PerceptronTrainer<'a, T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    type Model = PrototypeModel<T, N>;

    fn step(&mut self, epoch: usize) -> EpochResult {
        self.step(epoch)
    }

    fn into_model(self) -> Self::Model {
        self.into_model()
    }
}
