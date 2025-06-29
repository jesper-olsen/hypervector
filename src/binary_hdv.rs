use crate::HyperVector;
use rand::Rng;

impl<const N_USIZE: usize> HyperVector for BinaryHDV<N_USIZE> {
    fn new() -> Self {
        BinaryHDV::new()
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

const fn vec_size(dim: usize) -> usize {
    let bits_per_usize = usize::BITS as usize;
    let n = (dim + bits_per_usize - 1) / bits_per_usize;
    if n % 2 == 0 { n } else { n + 1 } // legacy - for balancing 1s & 0s on rng init
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate() {
        // note - if accumulating an even number of vectors, the result has a random component
        let mut v1 = BinaryHDV::<2>::zero();
        let mut v2 = BinaryHDV::<2>::zero();
        let mut v3 = BinaryHDV::<2>::zero();
        let mut r = BinaryHDV::<2>::zero();
        v1.data[0] = 5; // 0101
        v1.data[1] = 0;
        v2.data[0] = 1; // 0001
        v2.data[1] = 0;
        v3.data[0] = 9; // 1001
        v3.data[1] = 1;
        r.data[0] = 1;
        r.data[1] = 0;
        let b = BinaryHDV::<2>::acc(&[&v1, &v2, &v3]);
        assert_eq!(b, r);
    }

    #[test]
    fn binary_mexican_dollar() {
        crate::example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024 bits
    }
}
