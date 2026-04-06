use crate::trainer::{EpochResult, PrototypeModel, Trainer};
use crate::{Accumulator, HyperVector, nearest};
use rand::Rng;
use rand::prelude::SliceRandom;

/// Passive-Aggressive trainer, variants PA, PA-I, and PA-II.
///
/// See "Online Passive-Aggressive Algorithms",
///     Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, Yoram Singer,
///     Journal of Machine Learning Research 7 (2006) 551–585   
///     https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
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
    /// PA-I: step capped at C (e.g. 0.1). Larger C = more aggressive.
    PaI { c: f64 },
    /// PA-II: step softened by quadratic slack. Smoother than PA-I. Use e.g. c=1.0
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

pub struct PaTrainer<'a, T, L, R, const N: usize>
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
    variant: PaVariant,
}

impl<'a, T, L, R, const N: usize> PaTrainer<'a, T, L, R, N>
where
    T: HyperVector + Send + Sync,
    L: Into<usize> + Copy + Send + Sync,
    R: Rng,
{
    pub fn new(hvs: &'a [T], labels: &'a [L], variant: PaVariant, rng: R) -> Self {
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

        // TODO - batch training like PerceptronTrainer - allows parallel iteration
        let errors = self
            .indices
            .iter()
            .filter(|&&idx| {
                let hdv = &self.samples[idx];
                let true_class = self.labels[idx].into();
                let (predicted, _) = nearest(hdv, &self.prototypes);

                // Multiclass hinge loss: margin of true class vs best rival.
                // With cosine similarity this is bounded in [-2, 0] when violated.
                // Loss > 0 iff the margin constraint is not satisfied.
                let sim_true = hdv.similarity(&self.prototypes[true_class]);
                let sim_rival = hdv.similarity(&self.prototypes[predicted]);
                let loss = (1.0 - sim_true + sim_rival).max(0.0);

                if loss > 0.0 {
                    let norm_sq = 1.0; // true for binary, bipolar and normalised real/complex vectors ...
                    let tau = self.variant.tau(loss.into(), norm_sq);

                    self.accumulators[true_class].add(hdv, tau);
                    self.accumulators[predicted].add(hdv, -tau);

                    self.prototypes[true_class] = self.accumulators[true_class].finalize();
                    self.prototypes[predicted] = self.accumulators[predicted].finalize();
                }

                true_class != predicted
            })
            .count();

        EpochResult {
            epoch,
            correct: self.indices.len() - errors,
            errors,
        }
    }

    pub fn fit(mut self, max_epochs: usize) -> (PrototypeModel<T, N>, Vec<EpochResult>) {
        let mut history = Vec::with_capacity(max_epochs);
        for epoch in 1..=max_epochs {
            let result = self.step(epoch);
            history.push(result);
            if result.errors == 0 {
                // TODO - can still increase the margin...
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

impl<'a, T, L, R, const N: usize> Trainer<T> for PaTrainer<'a, T, L, R, N>
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
