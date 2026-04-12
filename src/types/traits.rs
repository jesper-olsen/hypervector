use rand::Rng;
use std::fs::File;

pub trait UnitAccumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T);
    fn finalize(&mut self) -> T;
    fn count(&self) -> usize;
}

pub trait Accumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T, weight: f64);
    fn finalize(&mut self) -> T;
    fn count(&self) -> f64;
}

pub trait HyperVector: Sized + Clone {
    type Accumulator: Default + Accumulator<Self>;
    type UnitAccumulator: Default + UnitAccumulator<Self>;
    const DIM: usize;

    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;
    /// Returns the identity element of the hypervector space:
    /// - Binary: all 0s (XOR identity)
    /// - Bipolar: all +1s (multiplicative identity)
    fn ident() -> Self;

    // blend two hypervectors by coping indices from other - rest from self
    fn blend(&self, other: &Self, indices: &[usize]) -> Self;

    fn distance(&self, other: &Self) -> f32; // 0..1
    fn similarity(&self, other: &Self) -> f32 {
        1.0 - self.distance(other)
    }

    fn bind(&self, other: &Self) -> Self;

    fn unbind(&self, other: &Self) -> Self;

    fn inverse(&self) -> Self;

    fn permute(&self, by: usize) -> Self;
    fn unpermute(&self, by: usize) -> Self;

    fn bundle(vectors: &[&Self]) -> Self {
        let mut acc: Self::UnitAccumulator = Self::UnitAccumulator::new();
        for v in vectors {
            acc.add(v)
        }
        acc.finalize()
    }

    fn unpack(&self) -> Vec<f32>;
    fn write(&self, file: &mut File) -> std::io::Result<()>;
    fn read(file: &mut File) -> std::io::Result<Self>;
}
