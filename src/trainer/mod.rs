use crate::{HyperVector, nearest};
use rayon::prelude::*;

pub mod kmeans;
pub mod lvq;
pub mod multi_perceptron;
pub mod pa;
pub mod perceptron;

pub trait Classifier<T: HyperVector> {
    fn predict(&self, h: &T) -> usize;

    /// returns tuple: number of correct, accuracy
    fn accuracy<L>(&self, samples: &[T], labels: &[L]) -> (usize, f64)
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
        (correct, acc)
    }
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
