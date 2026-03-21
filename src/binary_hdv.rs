use crate::{Accumulator, HyperVector, UnitAccumulator};
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::mem::size_of;

#[derive(Debug, PartialEq, Clone)]
pub struct BinaryHDV<const N_WORDS: usize> {
    pub data: [usize; N_WORDS],
}

impl<const N_WORDS: usize> HyperVector for BinaryHDV<N_WORDS> {
    type Accumulator = WeightedAcc<N_WORDS>;
    //type UnitAccumulator = UnitAcc<N_WORDS>;
    type UnitAccumulator = SlicedUnitAcc<N_WORDS>;
    const DIM: usize = N_WORDS * usize::BITS as usize;

    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let data = std::array::from_fn(|_| rng.next_u64() as usize);
        Self { data }
    }

    fn ident() -> Self {
        BinaryHDV { data: [0; N_WORDS] }
    }

    /// Creates a new HDV by blending `self` and `other`
    /// `indices` are bit positions where values from `other` are used.
    fn blend(&self, other: &Self, indices: &[usize]) -> Self {
        // Precompute masks per usize
        let mut masks = [0usize; N_WORDS];
        for &idx in indices {
            let i = idx / (usize::BITS as usize);
            let j = idx % (usize::BITS as usize);
            masks[i] |= 1 << j;
        }

        // Apply masks in bulk
        let data = std::array::from_fn(|i| (self.data[i] & !masks[i]) | (other.data[i] & masks[i]));
        Self { data }
    }

    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other) as f32 / (N_WORDS * usize::BITS as usize) as f32
    }

    fn bind(&self, other: &Self) -> Self {
        let data = std::array::from_fn(|i| self.data[i] ^ other.data[i]);
        Self { data }
    }

    fn unbind(&self, other: &Self) -> Self {
        Self::bind(self, other)
    }

    fn inverse(&self) -> Self {
        let data = self.data;
        Self { data }
    }

    //fn permute(&self, by: usize) -> Self {
    //    let mut result = Self::zero();
    //    for i in 0..N_WORDS {
    //        result.data[i] = self.data[(i + by) % N_WORDS];
    //    }
    //    result
    //}

    fn permute(&self, by: usize) -> Self {
        let mut result = Self::zero();
        let shift = by % Self::DIM;
        let word_shift = shift / usize::BITS as usize;
        let bit_shift = shift % usize::BITS as usize;

        for i in 0..N_WORDS {
            let target_idx = (i + word_shift) % N_WORDS;
            let next_idx = (target_idx + 1) % N_WORDS;

            // Carry bits from this word to the next to simulate a global circular shift
            result.data[target_idx] |= self.data[i] << bit_shift;
            if bit_shift > 0 {
                result.data[next_idx] |= self.data[i] >> (usize::BITS as usize - bit_shift);
            }
        }
        result
    }

    fn unpermute(&self, by: usize) -> Self {
        //self.permute(N_WORDS - (by % N_WORDS))
        self.permute(Self::DIM - (by % Self::DIM))
    }

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let data =
            std::array::from_fn(|i| self.data[(i + pa) % N_WORDS] ^ other.data[(i + pb) % N_WORDS]);
        Self { data }
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        Self::pbind(self, pa, other, pb)
    }

    fn unpack(&self) -> Vec<f32> {
        let n = N_WORDS * usize::BITS as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..N_WORDS {
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
        //     N_WORDS,
        //     N_WORDS as u32 * usize::BITS
        // );
        let mut data = [0usize; N_WORDS];
        for slot in &mut data {
            let mut buf = [0u8; size_of::<usize>()];
            file.read_exact(&mut buf)?;
            *slot = usize::from_ne_bytes(buf);
        }
        Ok(Self { data })
    }
}

// Consensus Accumulator
#[derive(Debug, Clone)]
pub struct WeightedAcc<const N_WORDS: usize> {
    votes: Vec<f64>, // one vote counter per bit
    count: f64,      // total number of vectors added
}

impl<const N_WORDS: usize> Default for WeightedAcc<N_WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_WORDS: usize> Accumulator<BinaryHDV<N_WORDS>> for WeightedAcc<N_WORDS> {
    fn new() -> Self {
        Self {
            votes: vec![0.0; N_WORDS * usize::BITS as usize],
            count: 0.0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_WORDS>, weight: f64) {
        for i in 0..N_WORDS {
            let word = v.data[i];
            for j in 0..usize::BITS {
                let flag = ((word >> j) & 1) as f64;
                self.votes[i * usize::BITS as usize + j as usize] += weight * flag
            }
        }
        self.count += weight;
    }

    fn finalize(&self) -> BinaryHDV<N_WORDS> {
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

    fn count(&self) -> f64 {
        self.count
    }
}

type VoteCount = u32;

#[derive(Debug, Clone)]
pub struct UnitAcc<const N_WORDS: usize> {
    votes: Vec<VoteCount>,
    count: usize, // total number of vectors added
}

impl<const N_WORDS: usize> Default for UnitAcc<N_WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_WORDS: usize> UnitAcc<N_WORDS> {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<const N_WORDS: usize> UnitAccumulator<BinaryHDV<N_WORDS>> for UnitAcc<N_WORDS> {
    fn new() -> Self {
        Self {
            votes: vec![0; N_WORDS * usize::BITS as usize],
            count: 0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_WORDS>) {
        for i in 0..N_WORDS {
            let word = v.data[i];
            for j in 0..usize::BITS {
                let flag = ((word >> j) & 1) as VoteCount;
                self.votes[i * usize::BITS as usize + j as usize] += flag;
            }
        }
        self.count += 1;
    }

    fn finalize(&self) -> BinaryHDV<N_WORDS> {
        // TODO - use bitslicing?
        let mut result = BinaryHDV::zero();
        let bits_per_word = usize::BITS as usize;

        for uidx in 0..N_WORDS {
            for bidx in 0..64 {
                let i = uidx * bits_per_word + bidx;
                let n1 = self.votes[i]; // #1s
                let n0 = (self.count - n1 as usize) as VoteCount; // #0s
                // TODO - generate a tie mask and only call random once per 64-bit word? Or just leave out the rand...
                if n1 > n0 || (n1 == n0 && rand::random::<bool>()) {
                    result.data[uidx] |= 1 << bidx;
                }
            }
        }

        result
    }

    fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug, Clone)]
pub struct SlicedUnitAcc<const N: usize> {
    // Same as UnitAcc, but implemented with "bit sliced counters" (aka parallel counters).
    // Layout: 32 contiguous bit-planes, each of length N
    // Plane 0 is at data[0..N], Plane 1 at data[N..2*N], etc.
    data: Vec<usize>,
    count: usize,
}

impl<const N_WORDS: usize> Default for SlicedUnitAcc<N_WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> UnitAccumulator<BinaryHDV<N>> for SlicedUnitAcc<N> {
    fn new() -> Self {
        Self {
            data: vec![0; N * 32],
            count: 0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N>) {
        // We use the input vector's data as the initial "carry"
        let mut carry_mask = v.data;

        for p_idx in 0..32 {
            let offset = p_idx * N;
            let mut any_carry = false;

            for i in 0..N {
                let p_val = self.data[offset + i];
                let c_in = carry_mask[i];

                // Half-adder logic applied to 64 bits at once
                self.data[offset + i] = p_val ^ c_in;
                carry_mask[i] = p_val & c_in;

                if carry_mask[i] > 0 {
                    any_carry = true;
                }
            }

            // If no bits are carrying over to the next plane, we can stop early
            if !any_carry {
                break;
            }
        }
        self.count += 1;
    }

    fn finalize(&self) -> BinaryHDV<N> {
        let mut result = BinaryHDV::zero();
        let threshold = self.count / 2;
        let is_even = self.count % 2 == 0;

        for i in 0..N {
            let mut word_result: usize = 0;

            for bit_idx in 0..usize::BITS {
                // Reconstruct the u32 count for this specific bit position
                let mut n1: u32 = 0;
                for p_idx in 0..32 {
                    let bit = (self.data[p_idx * N + i] >> bit_idx) & 1;
                    n1 |= (bit as u32) << p_idx;
                }

                if n1 > threshold as u32 {
                    word_result |= 1 << bit_idx;
                } else if is_even && n1 == threshold as u32 && rand::random::<bool>() {
                    // random may make little difference, but it is the correct way to handle ties
                    // computationally finalize is less important than add - assuming many adds
                    word_result |= 1 << bit_idx;
                }
            }
            result.data[i] = word_result;
        }

        result
    }

    fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug, Clone)]
pub struct GradientAccumulator<const N_WORDS: usize> {
    votes: Vec<f64>, // one vote counter per bit
    count: f64,      // total number of vectors added
}

impl<const N_WORDS: usize> Default for GradientAccumulator<N_WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_WORDS: usize> GradientAccumulator<N_WORDS> {
    pub const fn is_empty(&self) -> bool {
        self.count == 0.0
    }
}

impl<const N_WORDS: usize> Accumulator<BinaryHDV<N_WORDS>> for GradientAccumulator<N_WORDS> {
    fn new() -> Self {
        Self {
            votes: vec![0.0; N_WORDS * usize::BITS as usize],
            count: 0.0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_WORDS>, weight: f64) {
        for i in 0..N_WORDS {
            let word = v.data[i];
            for j in 0..usize::BITS {
                // MAP: Binary 1 -> 1.0, Binary 0 -> -1.0
                let bit_signal = if ((word >> j) & 1) == 1 { 1.0 } else { -1.0 };

                let idx = i * (usize::BITS as usize) + j as usize;
                self.votes[idx] += weight * bit_signal;
            }
        }
        self.count += weight.abs();
    }

    fn finalize(&self) -> BinaryHDV<N_WORDS> {
        let mut result = BinaryHDV::zero();
        for i in 0..self.votes.len() {
            // If the consensus is positive, set the bit to 1.
            // If it's 0.0, we flip a coin to avoid bias.
            if self.votes[i] > 0.0 || (self.votes[i] == 0.0 && rand::random::<bool>()) {
                let uidx = i / (usize::BITS as usize);
                let bidx = i % (usize::BITS as usize);
                result.data[uidx] |= 1 << bidx;
            }
        }
        result
    }

    fn count(&self) -> f64 {
        self.count
    }
}

impl<const N_WORDS: usize> BinaryHDV<N_WORDS> {
    pub fn from_slice(slice: &[u8]) -> Self {
        let dim = N_WORDS * usize::BITS as usize;
        assert!(slice.len() <= dim);
        let mut hdv = Self::zero();
        for (i, e) in slice.iter().enumerate() {
            let word_idx = i / usize::BITS as usize;
            let bit_idx = i % usize::BITS as usize;
            if *e != 0 {
                hdv.data[word_idx] |= 1 << bit_idx;
            }
        }
        hdv
    }

    pub fn zero() -> Self {
        Self {
            data: [0usize; N_WORDS],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&e| e == 0)
    }

    /// Returns a Vec<u8> with one entry per bit (0 or 1).
    pub fn as_u8_vec(&self) -> Vec<u8> {
        let mut bits = Vec::with_capacity(N_WORDS * usize::BITS as usize);
        for word in self.data {
            for i in 0..usize::BITS {
                let bit = (word >> i) & 1;
                bits.push(bit as u8);
            }
        }
        bits
    }

    /// Creates a new HDV by cloning `self` and flipping `nbits` unique, random bits.
    pub fn flip<R: Rng + ?Sized>(&self, nbits: usize, rng: &mut R) -> Self {
        let mut data = self.data;

        let dim = Self::DIM;
        if nbits == 0 || dim == 0 {
            return Self { data };
        }

        let mut flipped_indices = HashSet::new();
        let bits_to_flip = nbits.min(dim);

        while flipped_indices.len() < bits_to_flip {
            let idx = (rng.next_u64() as usize) % dim;

            if flipped_indices.insert(idx) {
                let i = idx / (usize::BITS as usize);
                let j = idx % (usize::BITS as usize);
                data[i] ^= 1 << j;
            }
        }

        Self { data }
    }

    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    pub fn xnor(&self, other: &Self) -> Self {
        let mut result = Self::zero();

        for i in 0..N_WORDS {
            // safe because DIM is a multiple of 64 - need to mask unused bits if this is not the case
            result.data[i] = !(self.data[i] ^ other.data[i]);
        }

        result
    }

    pub fn write_csv(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut buffer = Vec::with_capacity(Self::DIM * 2); // bits + commas
        for (i, word) in self.data.iter().enumerate() {
            for j in 0..usize::BITS {
                if i > 0 || j > 0 {
                    buffer.push(b',');
                }
                buffer.push(if (word >> j) & 1 == 1 { b'1' } else { b'0' });
            }
        }
        buffer.push(b'\n');
        writer.write_all(&buffer)
    }

    /// Render the hypervector as a Braille block of `width` characters per line.
    /// Each Braille char encodes 8 bits (2 cols × 4 rows, column-major).
    pub fn to_braille(&self, width: usize) -> String {
        let bits = self.as_u8_vec(); // already in order: word0 lsb first
        let total_chars = bits.len().div_ceil(8);
        let width = width.max(1);
        let height = total_chars.div_ceil(width);

        let mut out = String::with_capacity(height * (width * 3 + 1)); // UTF-8: braille = 3 bytes

        for row in 0..height {
            for col in 0..width {
                let char_idx = row * width + col;
                let bit_base = char_idx * 8;

                // Dot-to-bit mapping (column-major, standard Braille):
                // dot1=bit0, dot2=bit1, dot3=bit2, dot7=bit6  (left col)
                // dot4=bit3, dot5=bit4, dot6=bit5, dot8=bit7  (right col)
                let mut offset = 0u32;
                let dot_weights: [u32; 8] = [1, 2, 4, 64, 8, 16, 32, 128];
                for (k, &w) in dot_weights.iter().enumerate() {
                    let bit_idx = bit_base + k;
                    if bit_idx < bits.len() && bits[bit_idx] != 0 {
                        offset += w;
                    }
                }
                out.push(char::from_u32(0x2800 + offset).unwrap());
            }
            out.push('\n');
        }
        out
    }

    pub fn diff_braille(&self, other: &Self, width: usize) -> String {
        self.bind(other).to_braille(width)
    }
}

pub fn save_hdvs_to_csv<const N: usize>(
    filename: &str,
    hdv_dataset: &[BinaryHDV<N>],
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    for hdv in hdv_dataset {
        hdv.write_csv(&mut writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::binary_hdv::{BinaryHDV, UnitAcc, WeightedAcc};
    use crate::{Accumulator, HyperVector, UnitAccumulator};

    #[test]
    fn test_accumulate() {
        let v1 = BinaryHDV::<1>::from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
        let v2 = BinaryHDV::<1>::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let v3 = BinaryHDV::<1>::from_slice(&[1, 0, 0, 1, 0, 0, 0, 0]);
        let expected = BinaryHDV::<1>::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);

        let mut acc = WeightedAcc::default();
        acc.add(&v1, 1.0);
        acc.add(&v2, 1.0);
        acc.add(&v3, 1.0);
        assert_eq!(acc.finalize(), expected);

        let mut acc = UnitAcc::default();
        acc.add(&v1);
        acc.add(&v2);
        acc.add(&v3);
        assert_eq!(acc.finalize(), expected);

        let result = BinaryHDV::<1>::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }
}
