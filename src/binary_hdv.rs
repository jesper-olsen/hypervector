use crate::{Accumulator, HyperVector};
use rand::Rng;

impl<const N_USIZE: usize> HyperVector for BinaryHDV<N_USIZE> {
    type Accumulator = BinaryAccumulator<N_USIZE>;

    fn new() -> Self {
        BinaryHDV::new()
    }

    fn from_slice(slice: &[i8]) -> Self {
        let dim = N_USIZE * usize::BITS as usize;
        assert!(slice.len() <= dim);
        let mut hdv = Self::zero();
        for i in 0..slice.len() {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if slice[i] != 0 {
                hdv.data[word_idx] |= 1 << bit_idx;
            }
        }
        hdv
    }

    fn distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other) as f32 / (N_USIZE * usize::BITS as usize) as f32
    }

    fn multiply(&self, other: &Self) -> Self {
        BinaryHDV::multiply(self, other)
    }

    fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BinaryHDV::pmultiply(self, pa, other, pb)
    }

    fn acc(vectors: &[&Self]) -> Self {
        BinaryHDV::acc(vectors)
    }
}

#[derive(Debug, PartialEq)]
pub struct BinaryHDV<const N_USIZE: usize> {
    pub data: [usize; N_USIZE],
}

pub struct BinaryAccumulator<const N_USIZE: usize> {
    votes: Vec<usize>, // one vote counter per bit
    count: usize,      // total number of vectors added
}

impl<const N_USIZE: usize> Default for BinaryAccumulator<N_USIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_USIZE: usize> Accumulator<BinaryHDV<N_USIZE>> for BinaryAccumulator<N_USIZE> {
    fn new() -> Self {
        Self {
            votes: vec![0; N_USIZE * usize::BITS as usize],
            count: 0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_USIZE>) {
        for i in 0..N_USIZE {
            let word = v.data[i];
            for j in 0..usize::BITS {
                if (word >> j) & 1 != 0 {
                    self.votes[i * usize::BITS as usize + j as usize] += 1;
                }
            }
        }
        self.count += 1;
    }

    fn finalize(self) -> BinaryHDV<N_USIZE> {
        let mut result = BinaryHDV::zero();
        let bits_per_word = usize::BITS as usize;

        for i in 0..self.votes.len() {
            let uidx = i / bits_per_word;
            let bidx = i % bits_per_word;
            let n1 = self.votes[i];
            let n0 = self.count - n1;
            if n1 > n0 || (n1 == n0 && rand::random::<bool>()) {
                result.data[uidx] |= 1 << bidx;
            }
        }

        result
    }
}

impl<const N_USIZE: usize> BinaryHDV<N_USIZE> {
    fn new() -> Self {
        let mut rng = rand::rng();
        let data = std::array::from_fn(|_| rng.random_range(0..=usize::MAX));
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
    pub fn acc(vectors: &[&Self]) -> Self {
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

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_baccumulate() {
//         // note - if accumulating an even number of vectors, the result has a random component
//         let mut v1 = BinaryHDV::<2>::zero();
//         let mut v2 = BinaryHDV::<2>::zero();
//         let mut v3 = BinaryHDV::<2>::zero();
//         let mut r = BinaryHDV::<2>::zero();
//         v1.data[0] = 5; // 0101
//         v1.data[1] = 0;
//         v2.data[0] = 1; // 0001
//         v2.data[1] = 0;
//         v3.data[0] = 9; // 1001
//         v3.data[1] = 1;
//         r.data[0] = 1;
//         r.data[1] = 0;
//         let b = BinaryHDV::<2>::acc(&[&v1, &v2, &v3]);
//         assert_eq!(b, r);
//     }

//     #[test]
//     fn test_accumulate2() {
//         let mut acc = BinaryAccumulator::<2>::new();
//         // note - if accumulating an even number of vectors, the result has a random component
//         let v1 = BinaryHDV::<2>::from_slice(&[1, 0, 1, 0]);
//         let v2 = BinaryHDV::<2>::from_slice(&[1, 0, 0, 0]);
//         let v3 = BinaryHDV::<2>::from_slice(&[1, 0, 0, 1]);
//         let r = BinaryHDV::<2>::from_slice(&[1, 0, 0, 0]);

//         acc.add(&v1);
//         acc.add(&v2);
//         acc.add(&v3);
//         let b = acc.finalize();
//         assert_eq!(b, r);
//     }

//     #[test]
//     fn test_accumulate3() {
//         //test_accumulate::<BipolarHDV<5>>();
//         crate::test_accumulate::<BinaryHDV<64>>();
//     }

//     #[test]
//     fn binary_mexican_dollar() {
//         crate::example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024 bits
//     }
// }
