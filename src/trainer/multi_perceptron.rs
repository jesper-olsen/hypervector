use crate::trainer::{EpochResult, MultiPrototypeModel, Trainer, kmeans::KMeans};
use crate::{Accumulator, HyperVector, nearest};
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

/// Perceptron trainer for HDV multi prototype classifiers.
///
/// # Type parameters
/// - `T`: HyperVector type
/// - `R`: RNG (used for per-epoch shuffling)
/// - `L`: Label type — must be convertible to `usize` as a class index
pub struct PerceptronMultiTrainer<T, R>
where
    T: HyperVector + Send + Sync,
    R: Rng,
{
    accumulators: Vec<T::Accumulator>,
    prototypes: Vec<T>,
    samples: Vec<T>,
    proto_labels: Vec<usize>,
    n_classes: usize,
    proto_per_class: usize,
    indices: Vec<usize>,
    rng: R,
}

impl<T, R> PerceptronMultiTrainer<T, R>
where
    T: HyperVector + Send + Sync,
    R: Rng,
{
    pub fn new<L>(
        samples: Vec<T>,
        class_labels: Vec<L>,
        n_classes: usize,
        proto_per_class: usize,
        mut rng: R,
    ) -> Self
    where
        L: Into<usize> + Copy + Send + Sync,
    {
        let total_prototypes = n_classes * proto_per_class;
        let mut accumulators: Vec<T::Accumulator> = (0..total_prototypes)
            .map(|_| T::Accumulator::new())
            .collect();

        // Build kmeans per class to get initial prototypes
        let mut prototypes: Vec<T> = Vec::with_capacity(total_prototypes);
        for class in 0..n_classes {
            let class_samples: Vec<&T> = samples
                .iter()
                .zip(class_labels.iter().copied())
                .filter(|(_, l)| (*l).into() == class)
                .map(|(h, _)| h)
                .collect();
            let mut km = KMeans::new(&class_samples, proto_per_class, &mut rng);
            km.train(&class_samples, 100, false);
            prototypes.extend(km.centroids);
        }

        // Assign each sample to nearest prototype within its class
        // and compute new prototype-scoped label
        let mut proto_labels: Vec<usize> = Vec::with_capacity(samples.len());
        for (hdv, class_label) in samples.iter().zip(class_labels.iter()) {
            let class = (*class_label).into();
            let class_start = class * proto_per_class;
            let (nearest_idx, _) =
                nearest(hdv, &prototypes[class_start..class_start + proto_per_class]);
            let proto_label = class_start + nearest_idx;
            accumulators[proto_label].add(hdv, 1.0);
            proto_labels.push(proto_label);
        }

        let indices = (0..samples.len()).collect();
        Self {
            accumulators,
            prototypes,
            samples,
            proto_labels, // replaces original class labels for training
            n_classes,
            proto_per_class,
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
                let true_class = self.proto_labels[idx];
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

        self.prototypes = self.accumulators.iter_mut().map(|a| a.finalize()).collect();
        //self.prototypes = core::array::from_fn(|i| self.accumulators[i].finalize()).to_vec();

        EpochResult {
            epoch,
            correct: self.indices.len() - error_count,
            errors: error_count,
        }
    }

    /// Run up to `max_epochs` training steps, stopping early if zero errors.
    ///
    /// Returns the results of each epoch and the trained model.
    pub fn fit(mut self, max_epochs: usize) -> (MultiPrototypeModel<T>, Vec<EpochResult>) {
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

    /// Consume the trainer and return the final trained model.
    pub fn into_model(self) -> MultiPrototypeModel<T> {
        MultiPrototypeModel {
            prototypes: self.prototypes,
            proto_labels: self.proto_labels,
            n_classes: self.n_classes,
            proto_per_class: self.proto_per_class,
        }
    }
}

impl<T, R> Trainer<T> for PerceptronMultiTrainer<T, R>
where
    T: HyperVector + Send + Sync,
    R: Rng,
{
    type Model = MultiPrototypeModel<T>;
    fn step(&mut self, epoch: usize) -> EpochResult {
        self.step(epoch)
    }

    fn into_model(self) -> Self::Model {
        self.into_model()
    }
}
