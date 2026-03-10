// Modular Composite Representation", J. Snaider S. Franklin, 2014]
// https://digitalcommons.memphis.edu/ccrg_papers/32/
// Currently only the r=256 case

use crate::{Accumulator, HyperVector};
use rand_core::RngCore;
use std::fs::File;
use std::io::{Read, Write};

const R: u8 = 8; // bits per component (1-8), e.g. R=8 => MODULUS=256, HALF=128
const MODULUS: u32 = 1u32 << R;
const MASK: u8 = (MODULUS - 1) as u8;
const HALF: u32 = MODULUS >> 1;

// pre-compute lee distances
const LEE: [u8; MODULUS as usize] = {
    let mut table = [0u8; MODULUS as usize];
    let mut df = 0;
    while df < MODULUS {
        table[df as usize] = if df > HALF {
            (MODULUS - df) as u8
        } else {
            df as u8
        };
        df += 1;
    }
    table
};

use std::sync::OnceLock;

struct SinCosTables {
    sin: [f32; MODULUS as usize],
    cos: [f32; MODULUS as usize],
}

static SINCOS: OnceLock<SinCosTables> = OnceLock::new();

fn sincos_tables() -> &'static SinCosTables {
    SINCOS.get_or_init(|| {
        let mut sin = [0f32; MODULUS as usize];
        let mut cos = [0f32; MODULUS as usize];
        for i in 0..MODULUS as usize {
            let angle = (i as f32 / MODULUS as f32) * 2.0 * std::f32::consts::PI;
            sin[i] = angle.sin();
            cos[i] = angle.cos();
        }
        SinCosTables { sin, cos }
    })
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ModularHDV<const D: usize> {
    pub data: [u8; D],
}

impl<const DIM: usize> HyperVector for ModularHDV<DIM> {
    type Accumulator = ModularAccumulator<DIM>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        //let data = std::array::from_fn(|_| (rng.next_u32() & (MASK as u32)) as u8);
        let mut data = [0u8; DIM];
        rng.fill_bytes(&mut data);
        let data = data.map(|b| b & MASK);
        Self { data }
    }

    fn ident() -> Self {
        Self { data: [0u8; DIM] } // 0 is the additive identity for modulo arithmetic
    }

    fn distance(&self, other: &Self) -> f32 {
        // Max possible Lee distance is DIM * HALF
        // Normalise to [0.0; 1.0]
        self.lee_distance(other) as f32 / (DIM * HALF as usize) as f32
    }

    fn bind(&self, other: &Self) -> Self {
        let data: [u8; DIM] =
            std::array::from_fn(|i| self.data[i].wrapping_add(other.data[i]) & MASK);
        Self { data }
    }

    fn unbind(&self, other: &Self) -> Self {
        let data: [u8; DIM] =
            std::array::from_fn(|i| self.data[i].wrapping_sub(other.data[i]) & MASK);
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
        let p1 = pa % DIM;
        let p2 = pb % DIM;
        let data = std::array::from_fn(|i| {
            self.data[(i + p1) % DIM].wrapping_add(other.data[(i + p2) % DIM]) & MASK
        });
        Self { data }
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

    fn write(&self, file: &mut File) -> std::io::Result<()> {
        file.write_all(&self.data)
    }

    fn read(file: &mut File) -> std::io::Result<Self> {
        let mut data = [0u8; DIM];
        file.read_exact(&mut data)?;
        let data = data.map(|b| b & MASK); // Re-mask in case file was written with different R
        Ok(Self { data })
    }
}

#[derive(Clone)]
pub struct ModularAccumulator<const D: usize> {
    // We track sums of Sines and Cosines to find the circular mean
    sums_sin: [f32; D],
    sums_cos: [f32; D],
}

impl<const D: usize> Default for ModularAccumulator<D> {
    fn default() -> Self {
        ModularAccumulator::new()
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
        let t = sincos_tables();
        for i in 0..D {
            //let angle = (v.data[i] as f32 / 256.0) * 2.0 * std::f32::consts::PI;
            //self.sums_sin[i] += angle.sin();
            //self.sums_cos[i] += angle.cos();
            let idx = v.data[i] as usize;
            self.sums_sin[i] += t.sin[idx];
            self.sums_cos[i] += t.cos[idx];
        }
    }

    fn finalize(&self) -> ModularHDV<D> {
        let data = std::array::from_fn(|i| {
            if self.sums_sin[i].abs() < f32::EPSILON && self.sums_cos[i].abs() < f32::EPSILON {
                0
            } else {
                let angle = self.sums_sin[i].atan2(self.sums_cos[i]);
                // Normalize -PI..PI to 0.0..1.0
                let normalized = (angle / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);

                // map to the nearest discrete u8 gate
                let val = (normalized * MODULUS as f32).round() as u32;
                (val % MODULUS) as u8
            }
        });
        ModularHDV { data }
    }
}

impl<const DIM: usize> ModularHDV<DIM> {
    pub fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(slice.len(), DIM);
        let data = std::array::from_fn(|i| slice[i] & MASK); // all values in [0, MODULUS)
        Self { data }
    }

    // Lee distance: the shorter arc on the circle [0, MODULUS-1]
    // https://en.wikipedia.org/wiki/Lee_distance
    //fn lee_distance(&self, other: &Self) -> u32 {
    //    self.data
    //        .iter()
    //        .zip(other.data.iter())
    //        .map(|(x1, x2)| x1.wrapping_sub(*x2) as u32)
    //        .map(|df| if df > 128 { 256 - df } else { df })
    //        .sum()
    //}

    fn lee_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(u, v)| LEE[(u.wrapping_sub(*v) & MASK) as usize] as u32)
            .sum()
    }
}
