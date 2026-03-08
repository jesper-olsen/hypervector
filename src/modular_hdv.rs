// Modular Composite Representation", J. Snaider S. Franklin, 2014]
// https://digitalcommons.memphis.edu/ccrg_papers/32/
// Currently only the r=256 case

use crate::{Accumulator, HyperVector};
use rand_core::RngCore;
use std::array;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ModularHDV<const D: usize> {
    pub data: [u8; D],
}

impl<const DIM: usize> HyperVector for ModularHDV<DIM> {
    type Accumulator = ModularAccumulator<DIM>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let mut data = [0u8; DIM];
        rng.fill_bytes(&mut data);
        Self { data }
    }

    fn ident() -> Self {
        Self { data: [0u8; DIM] } // 0 is the additive identity for modulo arithmetic
    }

    fn distance(&self, other: &Self) -> f32 {
        self.lee_distance(other) as f32 / (DIM * 128) as f32 // Normalised to [0.0; 1.0] 
    }

    fn bind(&self, other: &Self) -> Self {
        let data: [u8; DIM] = array::from_fn(|i| self.data[i].wrapping_add(other.data[i]));
        Self { data }
    }

    fn unbind(&self, other: &Self) -> Self {
        let data: [u8; DIM] = array::from_fn(|i| self.data[i].wrapping_sub(other.data[i]));
        Self { data }
    }

    fn permute(&self, by: usize) -> Self {
        let mut result = [0u8; DIM];
        let shift = by % DIM;
        result[..shift].copy_from_slice(&self.data[DIM - shift..]);
        result[shift..].copy_from_slice(&self.data[..DIM - shift]);
        Self { data: result }
    }

    fn unpermute(&self, by: usize) -> Self {
        self.permute(DIM - (by % DIM))
    }

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let mut result = [0u8; DIM];
        let p1 = pa % DIM;
        let p2 = pb % DIM;
        for i in 0..DIM {
            result[i] = self.data[(i + p1) % DIM].wrapping_add(other.data[(i + p2) % DIM]);
        }
        Self { data: result }
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        self.pbind(pa, other, pb)
    }

    fn bundle(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            return Self::ident();
        }

        let mut acc = Self::Accumulator::new();
        for v in vectors {
            acc.add(v, 1.0);
        }
        acc.finalize()
    }

    fn unpack(&self) -> Vec<f32> {
        self.data.iter().map(|&e| e as f32).collect()
    }

    //fn write(&self, file: &mut File) -> std::io::Result<()> {
    //    unimplemented!()
    //}
    //fn read(file: &mut File) -> std::io::Result<Self> {
    //    unimplemented!()
    //}

    fn write(&self, file: &mut File) -> std::io::Result<()> {
        // Write all bytes at once
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, DIM) };
        file.write_all(bytes)
    }

    fn read(file: &mut File) -> std::io::Result<Self> {
        let mut data = [0u8; DIM];
        let buffer: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, DIM) };
        file.read_exact(buffer)?;
        Ok(Self { data })
    }
}

////#[derive(Debug, PartialEq, Clone, Copy)]
#[derive(Clone)]
pub struct ModularAccumulator<const D: usize> {
    // We track sums of Sines and Cosines to find the circular mean
    sums_sin: [f32; D],
    sums_cos: [f32; D],
}

impl<const DIM: usize> Default for ModularAccumulator<DIM> {
    fn default() -> Self {
        Self {
            sums_sin: [0.0; DIM],
            sums_cos: [0.0; DIM],
        }
    }
}

impl<const D: usize> Accumulator<ModularHDV<D>> for ModularAccumulator<D> {
    fn new() -> Self {
        Self {
            sums_sin: [0.0; D],
            sums_cos: [0.0; D],
        }
    }

    fn add(&mut self, v: &ModularHDV<D>, _weight: f64) {
        for i in 0..D {
            let angle = (v.data[i] as f32 / 256.0) * 2.0 * std::f32::consts::PI;
            self.sums_sin[i] += angle.sin();
            self.sums_cos[i] += angle.cos();
        }
    }

    //fn finalize(&self) -> ModularHDV<D> {
    //    let mut result = [0u8; D];
    //    for i in 0..D {
    //        let angle = self.sums_sin[i].atan2(self.sums_cos[i]);
    //        let normalized = (angle / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
    //        result[i] = (normalized * 256.0) as u8;
    //    }
    //    ModularHDV { data: result }
    //}

    fn finalize(&self) -> ModularHDV<D> {
        let mut result = [0u8; D];
        for i in 0..D {
            if self.sums_sin[i].abs() < f32::EPSILON && self.sums_cos[i].abs() < f32::EPSILON {
                result[i] = 0;
                continue;
            }

            let angle = self.sums_sin[i].atan2(self.sums_cos[i]);
            // Normalize -PI..PI to 0.0..1.0
            let normalized = (angle / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);

            // map to the nearest discrete u8 gate
            let val = (normalized * 256.0).round() as u32;
            result[i] = (val % 256) as u8;
        }
        ModularHDV { data: result }
    }
}

impl<const DIM: usize> ModularHDV<DIM> {
    pub fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(slice.len(), DIM);
        let data = std::array::from_fn(|i| slice[i]);
        ModularHDV { data }
    }

    // Lee distance: the shorter arc on the circle [0, 255]
    // https://en.wikipedia.org/wiki/Lee_distance
    fn lee_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..DIM {
            let diff = self.data[i].wrapping_sub(other.data[i]) as u32;
            dist += if diff > 128 {
                256 - (diff as u32)
            } else {
                diff as u32
            }
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use crate::modular_hdv::{ModularAccumulator, ModularHDV};
    use crate::{Accumulator, HyperVector};

    #[test]
    fn test_bind() {
        let v1 = ModularHDV::<2>::from_slice(&[100, 200]);
        let v2 = ModularHDV::<2>::from_slice(&[120, 240]);
        let v3 = v1.bind(&v2);
        assert_eq!(&v3.data, &[220, 184]);
    }

    #[test]
    fn test_accumulate() {
        let mut acc = ModularAccumulator::<5>::default();
        let v1 = ModularHDV::<5>::from_slice(&[1, 255, 1, 255, 255]);
        let v2 = ModularHDV::<5>::from_slice(&[1, 255, 255, 255, 255]);
        let v3 = ModularHDV::<5>::from_slice(&[1, 255, 255, 1, 255]);
        let expected = ModularHDV::<5>::from_slice(&[1, 255, 0, 0, 255]);

        acc.add(&v1, 1.0);
        acc.add(&v2, 1.0);
        acc.add(&v3, 1.0);
        let result = acc.finalize();
        assert_eq!(result, expected);

        let result = ModularHDV::<5>::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }
}
