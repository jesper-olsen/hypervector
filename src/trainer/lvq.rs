use crate::trainer::{EpochResult, MultiPrototypeModel, Trainer, kmeans::KMeans};
use crate::{Accumulator, HyperVector, nearest, nearest_two};
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

/// Perceptron trainer for HDV multi prototype classifiers.
///
/// # Type parameters
/// - `T`: HyperVector type
/// - `R`: RNG (used for per-epoch shuffling)
/// - `L`: Label type — must be convertible to `usize` as a class index
pub struct LvqTrainer<'a, T, R>
where
    T: HyperVector + Send + Sync,
    R: Rng,
{
    accumulators: Vec<T::Accumulator>,
    prototypes: Vec<T>,
    samples: &'a [T],
    proto_labels: Vec<usize>,
    n_classes: usize,
    proto_per_class: usize,
    indices: Vec<usize>,
    rng: R,
    window: f32, // typically from the range 0.2 - 0.3
}

impl<'a, T, R> LvqTrainer<'a, T, R>
where
    T: HyperVector + Send + Sync,
    R: Rng,
{
    pub fn new<L>(
        samples: &'a [T],
        class_labels: &'a [L],
        n_classes: usize,
        proto_per_class: usize,
        mut rng: R,
        window: f32,
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
            window,
        }
    }

    /// Run a single training epoch (LVQ2.1).
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
        let one_correct: Vec<(usize, usize, usize)> = self
            .indices
            .par_iter()
            .filter_map(|&idx| {
                let hdv = &self.samples[idx];
                let true_class = self.proto_labels[idx] / self.proto_per_class;

                let ((p1_idx, d1), (p2_idx, d2)) = nearest_two(hdv, &self.prototypes);
                let p1_class = p1_idx / self.proto_per_class;
                let p2_class = p2_idx / self.proto_per_class;
                // window condition: sample must be near the boundary between p1 and p2,
                // and exactly one of them must be the correct class
                let in_window = (d1 / d2).min(d2 / d1) > (1.0 - self.window) / (1.0 + self.window);
                let one_correct = (p1_class == true_class) != (p2_class == true_class);

                if in_window && one_correct {
                    // move correct prototype toward sample, wrong prototype away
                    let (correct_idx, wrong_idx) = if p1_class == true_class {
                        (p1_idx, p2_idx)
                    } else {
                        (p2_idx, p1_idx)
                    };
                    Some((idx, correct_idx, wrong_idx))
                } else {
                    None
                }
            })
            .collect();

        // Sequential: apply updates (accumulators not thread-safe)
        for (idx, correct_idx, wrong_idx) in one_correct {
            let hdv = &self.samples[idx];
            self.accumulators[correct_idx].add(hdv, lr);
            self.accumulators[wrong_idx].add(hdv, -lr);
        }

        self.prototypes = self.accumulators.iter_mut().map(|a| a.finalize()).collect();

        // Compute actual error count against all samples
        let errors: usize = self
            .indices
            .par_iter()
            .filter(|&&idx| {
                let hdv = &self.samples[idx];
                let true_class = self.proto_labels[idx] / self.proto_per_class;
                let (p1_idx, _) = nearest(hdv, &self.prototypes);
                p1_idx / self.proto_per_class != true_class
            })
            .count();

        EpochResult {
            epoch,
            correct: self.indices.len() - errors,
            errors,
        }
    }

    /// Run up to `max_epochs` training steps, stopping early if zero errors.
    /// Note that LVQ2.1 has no covergence guarantee and can fluctuate
    ///
    /// Returns the results of each epoch and the trained model.
    pub fn fit(mut self, max_epochs: usize) -> (MultiPrototypeModel<T>, Vec<EpochResult>) {
        let mut history = Vec::with_capacity(max_epochs);
        for epoch in 1..=max_epochs {
            let result = self.step(epoch);
            history.push(result);
        }
        (
            MultiPrototypeModel {
                prototypes: self.prototypes,
                proto_labels: self.proto_labels,
                n_classes: self.n_classes,
                proto_per_class: self.proto_per_class,
            },
            history,
        )
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

impl<'a, T, R> Trainer<T> for LvqTrainer<'a, T, R>
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
