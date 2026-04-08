use crate::{HyperVector, nearest};
use rayon::prelude::*;

pub mod har_dataset;
pub mod isolet_dataset;
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
