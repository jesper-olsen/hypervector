use crate::{Accumulator, HyperVector};
use rand_core::RngCore;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::size_of;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BinaryHDV<const N_USIZE: usize> {
    pub data: [usize; N_USIZE],
}

impl<const N_USIZE: usize> HyperVector for BinaryHDV<N_USIZE> {
    type Accumulator = BinaryAccumulator<N_USIZE>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        BinaryHDV::random(rng)
    }

    fn ident() -> Self {
        BinaryHDV { data: [0; N_USIZE] }
    }

    fn from_slice(slice: &[f32]) -> Self {
        let dim = N_USIZE * usize::BITS as usize;
        assert!(slice.len() <= dim);
        let mut hdv = Self::zero();
        for i in 0..slice.len() {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if slice[i] > 0.5 {
                hdv.data[word_idx] |= 1 << bit_idx;
            }
        }
        hdv
    }

    fn distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other) as f32 / (N_USIZE * usize::BITS as usize) as f32
    }

    fn bind(&self, other: &Self) -> Self {
        BinaryHDV::multiply(self, other)
    }

    fn unbind(&self, other: &Self) -> Self {
        BinaryHDV::multiply(self, other)
    }

    fn permute(&self, by: usize) -> Self {
        let mut result = Self::zero();
        for i in 0..N_USIZE {
            result.data[i] = self.data[(i + by) % N_USIZE];
        }
        result
    }

    fn unpermute(&self, by: usize) -> Self {
        self.permute(N_USIZE - (by % N_USIZE))
    }

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BinaryHDV::pmultiply(self, pa, other, pb)
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BinaryHDV::pmultiply(self, pa, other, pb)
    }

    fn bundle(vectors: &[&Self]) -> Self {
        BinaryHDV::bundle(vectors)
    }

    fn unpack(&self) -> Vec<f32> {
        let n = N_USIZE * usize::BITS as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..N_USIZE {
            for j in 0..usize::BITS {
                let bit = (self.data[i] >> j) & 1;
                out.push(if bit == 1 { 1.0 } else { 0.0 });
            }
        }
        out
    }

    fn write(&self, file: &mut File) -> io::Result<()> {
        for &value in &self.data {
            file.write_all(&value.to_ne_bytes())?;
        }
        Ok(())
    }

    fn read(file: &mut File) -> io::Result<Self> {
        // println!(
        //     "Reading HDV of {} usize elements = {} bits",
        //     N_USIZE,
        //     N_USIZE as u32 * usize::BITS
        // );
        let mut data = [0usize; N_USIZE];
        for slot in &mut data {
            let mut buf = [0u8; size_of::<usize>()];
            file.read_exact(&mut buf)?;
            *slot = usize::from_ne_bytes(buf);
        }
        Ok(Self { data })
    }
}

#[derive(Debug, Clone)]
pub struct BinaryAccumulator<const N_USIZE: usize> {
    votes: Vec<f64>, // one vote counter per bit
    count: f64,      // total number of vectors added
}

impl<const N_USIZE: usize> Default for BinaryAccumulator<N_USIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_USIZE: usize> Accumulator<BinaryHDV<N_USIZE>> for BinaryAccumulator<N_USIZE> {
    fn new() -> Self {
        Self {
            votes: vec![0.0; N_USIZE * usize::BITS as usize],
            count: 0.0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_USIZE>, weight: f64) {
        for i in 0..N_USIZE {
            let word = v.data[i];
            for j in 0..usize::BITS {
                let flag = ((word >> j) & 1) as f64;
                self.votes[i * usize::BITS as usize + j as usize] += weight * flag
            }
        }
        self.count += weight;
    }

    fn finalize(&self) -> BinaryHDV<N_USIZE> {
        let mut result = BinaryHDV::zero();
        let bits_per_word = usize::BITS as usize;

        for i in 0..self.votes.len() {
            let uidx = i / bits_per_word;
            let bidx = i % bits_per_word;
            let n1 = self.votes[i]; // #1s
            let n0 = self.count - n1; // #0s
            if n1 > n0 || (n1 == n0 && rand::random::<bool>()) {
                result.data[uidx] |= 1 << bidx;
            }
        }

        result
    }
}

impl<const N_USIZE: usize> BinaryHDV<N_USIZE> {
    pub fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let data = std::array::from_fn(|_| rng.next_u64() as usize);
        Self { data }
    }

    pub fn zero() -> Self {
        Self {
            data: [0usize; N_USIZE],
        }
    }

    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as usize)
            .sum()
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = Self::zero();

        for i in 0..N_USIZE {
            result.data[i] = self.data[i] ^ other.data[i];
        }

        result
    }

    /// Like multiply, but permute vectors first
    pub fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let mut result = Self::zero();

        for i in 0..N_USIZE {
            result.data[i] = self.data[(i + pa) % N_USIZE] ^ other.data[(i + pb) % N_USIZE];
        }

        result
    }

    /// count number of 1 bits for each bit position - if more than half are 1, then set
    /// that bit position to 1 in the returned vector
    /// For ties: random 1 or 0
    pub fn bundle(vectors: &[&Self]) -> Self {
        const BITS_PER_USIZE: usize = usize::BITS as usize;
        let mut a = vec![0usize; N_USIZE * 64]; // vote counter per bit
        let mut r = Self::zero(); // result vector

        for i in 0..N_USIZE {
            for j in 0..BITS_PER_USIZE {
                for v in vectors {
                    a[i * BITS_PER_USIZE + j] += (v.data[i] >> j) & 1usize;
                }
            }
        }

        for i in 0..N_USIZE * BITS_PER_USIZE {
            let uidx = i / BITS_PER_USIZE;
            let bidx = i % BITS_PER_USIZE;
            let a0 = vectors.len() - a[i];
            if a[i] > a0 || (a[i] == a0 && rand::random::<bool>()) {
                r.data[uidx] |= 1usize << bidx;
            }
        }

        r
    }
}
