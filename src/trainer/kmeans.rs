use crate::nearest;
use crate::types::traits::{HyperVector, UnitAccumulator};
use rand::Rng;
use rand::prelude::IndexedRandom;
use std::borrow::Borrow;

pub struct KMeans<H: HyperVector> {
    pub k: usize,
    pub centroids: Vec<H>,
    pub counts: Vec<usize>,
}

impl<H: HyperVector> KMeans<H> {
    /// Creates a new KMeans model with initial centroids chosen randomly.
    pub fn new<T: Borrow<H>, R: Rng>(data: &[T], k: usize, rng: &mut R) -> Self {
        // Borrow means the function accepts slice of either HyperVector or reference to HyperVector...
        assert!(k > 0 && k <= data.len(), "k must be > 0 and <= data length");

        // Simple random initialization.
        // TODO: implement K-Means++ initialisation.
        let centroids: Vec<H> = data.sample(rng, k).map(|v| v.borrow().clone()).collect();
        Self {
            k,
            centroids,
            counts: vec![0; k],
        }
    }

    /// Trains the model until convergence or max_iters is reached.
    pub fn train<T: Borrow<H>>(&mut self, data: &[T], max_iters: u32, verbose: bool) -> usize {
        let mut last_total_dist = usize::MAX;
        if verbose {
            println!("Cluster {} examples", data.len());
        }
        for i in 1..=max_iters {
            let total_dist = self.step(data);
            if verbose {
                println!("Iteration {i:3}: total distance = {total_dist:11}");
            }
            if last_total_dist == total_dist {
                if verbose {
                    println!("Converged after {i} iterations.");
                }
                break;
            }
            last_total_dist = total_dist;
        }
        last_total_dist
    }

    fn step<T: Borrow<H>>(&mut self, data: &[T]) -> usize {
        let mut accumulators: Vec<H::UnitAccumulator> =
            (0..self.k).map(|_| H::UnitAccumulator::default()).collect();

        let total_dist: usize = data
            .iter()
            .map(|v| {
                let (idx, dist) = self.nearest(v.borrow());
                accumulators[idx].add(v.borrow());
                dist as usize
            })
            .sum();

        self.centroids = accumulators.iter_mut().map(|a| a.finalize()).collect();
        self.counts = accumulators.iter().map(|a| a.count()).collect();
        total_dist
    }

    /// Finds the index and distance of the nearest centroid to a given vector.
    pub fn nearest(&self, hdv: &H) -> (usize, f32) {
        nearest(hdv, &self.centroids)
    }
}
