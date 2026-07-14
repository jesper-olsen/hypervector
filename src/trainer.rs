use crate::{HyperVector, nearest};
use rayon::prelude::*;

pub mod kmeans;
pub mod lvq;
pub mod multi_perceptron;
pub mod pa;
pub mod perceptron;

pub trait Classifier<T: HyperVector> {
    fn predict(&self, h: &T) -> usize;

    fn classify_all(&self, samples: &[T]) -> Vec<usize>
    where
        T: Send + Sync,
        Self: Sync,
    {
        samples.par_iter().map(|h| self.predict(h)).collect()
    }

    /// returns tuple: number of correct, errors, accuracy
    fn accuracy<L>(&self, samples: &[T], labels: &[L]) -> (usize, usize, f64)
    where
        L: Into<usize> + Copy + Send + Sync,
        T: Send + Sync,
        Self: Sync,
    {
        assert!(!samples.is_empty() && samples.len() == labels.len());
        let correct: usize = samples
            .par_iter()
            .zip(labels.par_iter().copied())
            .filter(|(h, label)| self.predict(h) == (*label).into())
            .count();
        let acc = correct as f64 / samples.len() as f64;
        (correct, samples.len() - correct, acc)
    }

    //fn classify<L>(&self, samples: &[T]) -> Vec<usize>
    //where
    //    T: Send + Sync,
    //    Self: Sync,
    //{
    //    samples
    //        .par_iter()
    //        .zip(labels.par_iter().copied())
    //        .map(|(h, label)| self.predict(h))
    //        .collect()
    //}
}

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
    pub fn scores(&self, h: &T) -> [f32; N] {
        std::array::from_fn(|i| self.prototypes[i].distance(&h))
    }
}

impl<T: HyperVector, const N: usize> Classifier<T> for PrototypeModel<T, N> {
    fn predict(&self, h: &T) -> usize {
        let (idx, _) = nearest(h, &self.prototypes);
        idx
    }
}

pub struct MultiPrototypeModel<T: HyperVector> {
    pub prototypes: Vec<T>,
    pub proto_labels: Vec<usize>, // class for each prototype, len = n_classes * proto_per_class
    pub n_classes: usize,
    pub proto_per_class: usize,
}

impl<T: HyperVector> Classifier<T> for MultiPrototypeModel<T> {
    fn predict(&self, h: &T) -> usize {
        let (idx, _) = nearest(h, &self.prototypes);
        idx / self.proto_per_class
    }
}

pub trait Trainer<T: HyperVector> {
    type Model: Classifier<T>;
    fn step(&mut self, epoch: usize) -> EpochResult;

    /// Consume the trainer and return the final trained model.
    fn into_model(self) -> Self::Model;
}

pub fn ensemble_vote<L>(predictions: &[Vec<usize>], n_classes: usize) -> Vec<usize> {
    let n_samples = predictions[0].len();
    (0..n_samples)
        .map(|i| {
            let mut votes = vec![0usize; n_classes];
            for model_preds in predictions {
                votes[model_preds[i]] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, v)| v)
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect()
}

pub fn ensemble_accuracy<L>(
    predictions: &[Vec<usize>],
    labels: &[L],
    n_classes: usize,
) -> (usize, usize, f64)
where
    L: Into<usize> + Copy,
{
    let votes = ensemble_vote::<L>(predictions, n_classes);
    let correct = votes
        .iter()
        .zip(labels.iter())
        .filter(|(pred, label)| **pred == (**label).into())
        .count();
    let total = labels.len();
    (correct, total - correct, correct as f64 / total as f64)
}

/// Sum-rule fusion for score-based ensembles.
///
/// `scores` holds one `Vec<[f32; N]>` per model, each indexed by sample.
/// Each entry is that model's per-class distance for the sample (lower = closer).
/// The fused prediction is the class with the smallest *summed* distance
/// across all models. Valid to sum directly (no per-model normalization)
/// as long as every model shares the same encoder/dimensionality, so their
/// distance scales are already comparable.
pub fn ensemble_fusion<const N: usize>(scores: &[Vec<[f32; N]>]) -> Vec<usize> {
    let n_samples = scores[0].len();
    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut summed = [0f32; N];
            for model_scores in scores {
                for c in 0..N {
                    summed[c] += model_scores[i][c];
                }
            }
            argmin(&summed)
            //summed
            //    .iter()
            //    .enumerate()
            //    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            //    .map(|(idx, _)| idx)
            //    .unwrap()
        })
        .collect()
}

pub fn argmin<const N: usize>(scores: &[f32; N]) -> usize {
    let mut min_idx = 0;
    let mut min_val = scores[0];
    for i in 1..N {
        if scores[i] < min_val {
            min_val = scores[i];
            min_idx = i;
        }
    }
    min_idx
}

pub fn ensemble_fusion_accuracy<L, const N: usize>(
    scores: &[Vec<[f32; N]>],
    labels: &[L],
) -> (usize, usize, f64)
where
    L: Into<usize> + Copy,
{
    let preds = ensemble_fusion(scores);
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|(pred, label)| **pred == (**label).into())
        .count();
    let total = labels.len();
    (correct, total - correct, correct as f64 / total as f64)
}
