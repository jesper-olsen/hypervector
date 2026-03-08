// BipolarHDV - note that this module uses one i8 per dimension.
// It can be implemented more compactly with bitarrays.
// However, that would make the implementation basically the same as BinaryHDV.
// Only difference is how 0 and 1 are interpreted - ie. for bipolar 0=1 and 1=-1.

use crate::{Accumulator, HyperVector};
use rand::Rng;
use rand_core::RngCore;
use std::fs::File;
use std::io::{Read, Write};

impl<const DIM: usize> HyperVector for BipolarHDV<DIM> {
    type Accumulator = BipolarAccumulator<DIM>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        BipolarHDV::random(rng)
    }

    fn ident() -> Self {
        BipolarHDV { data: [1; DIM] }
    }

    fn distance(&self, other: &Self) -> f32 {
        // 1.0 - cos angle between self and other
        // = 1.0 - dot(self,other)/(norm(self)*norm(other)
        // for bipolar vectors, the norm is sqrt(DIM).
        // distance is a number in the interval 0..2
        let dot: i32 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a * b) as i32)
            .sum();
        1.0 - (dot as f32) / DIM as f32
    }

    fn bind(&self, other: &Self) -> Self {
        BipolarHDV::multiply(self, other)
    }

    fn unbind(&self, other: &Self) -> Self {
        BipolarHDV::multiply(self, other)
    }

    fn permute(&self, by: usize) -> Self {
        let data = std::array::from_fn(|i| self.data[(i + by) % DIM]);
        Self { data }
    }

    fn unpermute(&self, by: usize) -> Self {
        self.permute(DIM - (by % DIM))
    }

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BipolarHDV::pmultiply(self, pa, other, pb)
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BipolarHDV::pmultiply(self, pa, other, pb)
    }

    fn bundle(vectors: &[&Self]) -> Self {
        BipolarHDV::bundle(vectors)
    }

    fn unpack(&self) -> Vec<f32> {
        self.data.iter().map(|&e| e as f32).collect()
    }

    fn write(&self, file: &mut File) -> std::io::Result<()> {
        // Write all bytes at once
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, DIM) };
        file.write_all(bytes)
    }

    fn read(file: &mut File) -> std::io::Result<Self> {
        let mut data = [0i8; DIM];
        let buffer: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, DIM) };
        file.read_exact(buffer)?;
        Ok(Self { data })
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BipolarAccumulator<const DIM: usize> {
    sum: [f64; DIM],
}

impl<const DIM: usize> Default for BipolarAccumulator<DIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const DIM: usize> Accumulator<BipolarHDV<DIM>> for BipolarAccumulator<DIM> {
    fn new() -> Self {
        Self { sum: [0.0; DIM] }
    }

    fn add(&mut self, v: &BipolarHDV<DIM>, weight: f64) {
        for i in 0..DIM {
            self.sum[i] += weight * v.data[i] as f64;
        }
    }

    fn finalize(&self) -> BipolarHDV<DIM> {
        let data = std::array::from_fn(|i| match self.sum[i] {
            s if s > 0.0 => 1,
            s if s < 0.0 => -1,
            _ if rand::random() => 1,
            _ => -1,
        });
        BipolarHDV { data }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BipolarHDV<const DIM: usize> {
    data: [i8; DIM], // +1 or -1
}

impl<const DIM: usize> BipolarHDV<DIM> {
    pub fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let data = std::array::from_fn(|_| if rng.random_bool(0.5) { 1 } else { -1 });
        Self { data }
    }

    pub fn from_slice(slice: &[i8]) -> Self {
        assert_eq!(slice.len(), DIM);
        let data = std::array::from_fn(|i| if slice[i] == 1 { 1 } else { -1 });
        BipolarHDV { data }
    }

    /// sum HDVs in l self and normalise
    pub fn bundle(l: &[&BipolarHDV<DIM>]) -> Self {
        let data = std::array::from_fn(|i| {
            let s: i64 = l.iter().map(|v| v.data[i] as i64).sum();
            match s {
                _ if s > 0 => 1,
                _ if s < 0 => -1,
                _ if rand::random() => 1,
                _ => -1,
            }
        });
        Self { data }
    }

    pub fn multiply(&self, b: &Self) -> Self {
        let data = std::array::from_fn(|i| self.data[i] * b.data[i]);
        Self { data }
    }

    /// permute HDVs then multiply
    pub fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let p1 = pa % DIM;
        let p2 = pb % DIM;
        let data = std::array::from_fn(|i| self.data[(i + p1) % DIM] * other.data[(i + p2) % DIM]);
        Self { data }
    }
}

#[cfg(test)]
mod tests {
    use crate::bipolar_hdv::{BipolarAccumulator, BipolarHDV};
    use crate::Accumulator;

    #[test]
    fn test_accumulate() {
        let mut acc = BipolarAccumulator::<5>::default();
        let v1 = BipolarHDV::<5>::from_slice(&[1, -1, 1, -1, -1]);
        let v2 = BipolarHDV::<5>::from_slice(&[1, -1, -1, -1, -1]);
        let v3 = BipolarHDV::<5>::from_slice(&[1, -1, -1, 1, -1]);
        let expected = BipolarHDV::<5>::from_slice(&[1, -1, -1, -1, -1]);

        acc.add(&v1, 1.0);
        acc.add(&v2, 1.0);
        acc.add(&v3, 1.0);
        let result = acc.finalize();
        assert_eq!(result, expected);

        let result = BipolarHDV::<5>::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }
}
