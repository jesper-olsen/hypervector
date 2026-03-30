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
///
/// # Example
/// ```rust
/// let trainer = PerceptronTrainer::<HDV, u8, _, 10>::new(
///     samples.into_iter().map(|(img_hdv, label)| (img_hdv, label)),
///     StdRng::seed_from_u64(42),
/// );
/// let model = trainer.fit(100);
/// ```
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

/// Passive-Aggressive trainer, variants PA, PA-I, and PA-II.
///
/// Implements Figure 1 / Section 7 of Crammer et al. (2006), adapted for
/// multiclass: on a margin violation, add τ·x to the true-class accumulator
/// and subtract τ·x from the rival accumulator.
///
/// Unlike Pegasos there is no shrinkage — non-violating rounds are fully
/// passive. This makes all three variants compatible with binary, bipolar,
/// modular, real and complex HDVs.
#[derive(Debug, Clone, Copy)]
pub enum PaVariant {
    /// PA: unconstrained step. No hyperparameter. Can be large on outliers.
    Pa,
    /// PA-I: step capped at C. Larger C = more aggressive.
    PaI { c: f64 },
    /// PA-II: step softened by quadratic slack. Smoother than PA-I.
    PaII { c: f64 },
}

impl PaVariant {
    fn tau(&self, loss: f64, norm_sq: f64) -> f64 {
        match self {
            PaVariant::Pa => loss / norm_sq,
            PaVariant::PaI { c } => (loss / norm_sq).min(*c),
            PaVariant::PaII { c } => loss / (norm_sq + 1.0 / (2.0 * c)),
        }
    }
}

pub struct PaTrainer<T, L, R, const N: usize>
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
    variant: PaVariant,
}

impl<T, L, R, const N: usize> PaTrainer<T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    pub fn new(hvs: Vec<T>, labels: Vec<L>, variant: PaVariant, rng: R) -> Self {
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
            samples: hvs,
            labels,
            indices: (0..n).collect(),
            rng,
            variant,
        }
    }

    /// One PA epoch. For each sample in shuffled order:
    ///
    ///   loss  = max(0, 1 - similarity(w_true, x) + similarity(w_rival, x))
    ///   τ     = variant.tau(loss, ‖x‖²)
    ///   if loss > 0:
    ///       w_true  ← w_true  + τ · x
    ///       w_rival ← w_rival − τ · x
    ///
    /// Passive on correct margin-satisfying predictions — w is unchanged.
    pub fn step(&mut self, epoch: usize) -> EpochResult {
        self.indices.shuffle(&mut self.rng);
        let mut error_count = 0;

        for &idx in &self.indices {
            let hdv = &self.samples[idx];
            let true_class = self.labels[idx].into();
            let (predicted, _) = nearest(hdv, &self.prototypes);

            // Multiclass hinge loss: margin of true class vs best rival.
            // With cosine similarity this is bounded in [-2, 0] when violated.
            // Loss > 0 iff the margin constraint is not satisfied.

            let margin = 0.1;
            //let sim_true = 1.0 - hdv.distance(&self.prototypes[true_class]);
            //let sim_rival = 1.0 - hdv.distance(&self.prototypes[predicted]);
            //let loss = (1.0 - sim_true + sim_rival).max(0.0);

            // violation if dist_true > dist_rival + margin
            let dist_true  = hdv.distance(&self.prototypes[true_class]) as f64;
            let dist_rival = hdv.distance(&self.prototypes[predicted]) as f64;
            let loss = (dist_true - dist_rival + margin).max(0.0);


            if loss > 0.0 {
                let norm_sq = 1.0; // TODO - true for binary and bipolar HDVs
                let tau = self.variant.tau(loss.into(), norm_sq);

                self.accumulators[true_class].add(hdv, tau);
                self.accumulators[predicted].add(hdv, -tau);

                self.prototypes[true_class] = self.accumulators[true_class].finalize();
                self.prototypes[predicted] = self.accumulators[predicted].finalize();

                error_count += 1;
            }
        }

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
        (
            PrototypeModel {
                prototypes: self.prototypes,
            },
            history,
        )
    }

    pub fn prototypes(&self) -> &[T; N] {
        &self.prototypes
    }

    pub fn into_model(self) -> PrototypeModel<T, N> {
        PrototypeModel {
            prototypes: self.prototypes,
        }
    }
}
