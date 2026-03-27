use crate::{Accumulator, HyperVector, UnitAccumulator};
use mersenne_twister_rs::MersenneTwister64;
use rand::{Rng, SeedableRng};
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
    type UnitAccumulator = SlicedUnitAcc<N_WORDS, 32>; // 1-64 bit PLANES
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

    fn permute(&self, by: usize) -> Self {
        self.permute_idx(by)
        //self.permute_bit(by)
    }

    fn unpermute(&self, by: usize) -> Self {
        let shift = by % Self::DIM;
        // Moving backwards 'shift' is the same as moving forwards 'DIM - shift'
        self.permute(Self::DIM - shift)
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

#[derive(Debug, Clone)]
pub struct WeightedAcc<const N_WORDS: usize, R: Rng = MersenneTwister64> {
    votes: [[f64; usize::BITS as usize]; N_WORDS], // one vote counter per bit
    count: f64,                                    // total number of vectors added
    rng: R,
}

impl<const N_WORDS: usize, R: Rng + SeedableRng + Default> Default for WeightedAcc<N_WORDS, R> {
    fn default() -> Self {
        Self {
            votes: [[0.0; usize::BITS as usize]; N_WORDS],
            count: 0.0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }
}

impl<const N_WORDS: usize> Accumulator<BinaryHDV<N_WORDS>> for WeightedAcc<N_WORDS> {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, v: &BinaryHDV<N_WORDS>, weight: f64) {
        for i in 0..N_WORDS {
            let word = v.data[i];
            for j in 0..usize::BITS as usize {
                if (word >> j) & 1 == 1 {
                    self.votes[i][j] += weight
                }
            }
        }
        self.count += weight.abs();
    }

    fn finalize(&mut self) -> BinaryHDV<N_WORDS> {
        let data = std::array::from_fn(|uidx| {
            let mut word_acc = 0usize;
            let tie_breaker: usize = self.rng.next_u64() as usize;
            for bidx in 0..usize::BITS as usize {
                let n1 = self.votes[uidx][bidx];
                let n0 = self.count - n1;
                if n1 > n0 || (n1 == n0 && (tie_breaker & (1 << bidx)) != 0) {
                    word_acc |= 1 << bidx;
                }
            }
            word_acc
        });
        BinaryHDV::<N_WORDS> { data }
    }

    fn count(&self) -> f64 {
        self.count
    }
}

type VoteCount = u32;

#[derive(Debug, Clone)]
pub struct UnitAcc<const N_WORDS: usize> {
    votes: [[VoteCount; usize::BITS as usize]; N_WORDS],
    count: usize, // total number of vectors added
}

impl<const N_WORDS: usize> Default for UnitAcc<N_WORDS> {
    fn default() -> Self {
        Self {
            votes: [[0; usize::BITS as usize]; N_WORDS],
            count: 0,
        }
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
            votes: [[0; usize::BITS as usize]; N_WORDS],
            count: 0,
        }
    }

    fn add(&mut self, v: &BinaryHDV<N_WORDS>) {
        for i in 0..N_WORDS {
            let word = v.data[i];
            for j in 0..usize::BITS {
                let flag = ((word >> j) & 1) as VoteCount;
                self.votes[i][j as usize] += flag;
            }
        }
        self.count += 1;
    }

    fn finalize(&mut self) -> BinaryHDV<N_WORDS> {
        // TODO - use bitslicing?
        let mut result = BinaryHDV::zero();

        for uidx in 0..N_WORDS {
            for bidx in 0..usize::BITS as usize {
                let n1 = self.votes[uidx][bidx]; // #1s
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

pub struct SlicedUnitAcc<const N: usize, const PLANES: usize, R: Rng = MersenneTwister64> {
    // Each plane is a bit-level of the parallel counters.
    // Plane 0 = Least Significant Bit, Plane PLANES-1 = Most Significant Bit.
    data: [[usize; N]; PLANES],
    count: usize,
    rng: R,
}

impl<const N_WORDS: usize, const PLANES: usize, R: Rng + SeedableRng + Default> Default
    for SlicedUnitAcc<N_WORDS, PLANES, R>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const PLANES: usize, R: Rng + SeedableRng + Default>
    UnitAccumulator<BinaryHDV<N>> for SlicedUnitAcc<N, PLANES, R>
{
    fn new() -> Self {
        Self {
            data: [[0; N]; PLANES],
            count: 0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }

    #[inline]
    fn add(&mut self, v: &BinaryHDV<N>) {
        let mut carry = v.data;

        for p in 0..PLANES {
            for (old_val_ref, carry_word) in self.data[p].iter_mut().zip(carry.iter_mut()) {
                let old_val = *old_val_ref;
                *old_val_ref = old_val ^ *carry_word;
                *carry_word &= old_val;
            }
            if carry.iter().all(|&c| c == 0) {
                break;
            }
        }
        self.count += 1;
    }

    fn finalize(&mut self) -> BinaryHDV<N> {
        let mut result = BinaryHDV::<N>::zero();
        let threshold = (self.count / 2) as u64;
        let is_even = self.count.is_multiple_of(2);

        for i in 0..N {
            let tie_breaker: usize = self.rng.next_u64() as usize;
            let mut word_acc = 0usize;
            // We process all 64 bits of the word simultaneously for each bit-position
            for bit_pos in 0..usize::BITS as usize {
                let mut bit_count = 0u64;
                for p in 0..PLANES {
                    let bit = (self.data[p][i] >> bit_pos) & 1;
                    bit_count |= (bit as u64) << p;
                }

                if bit_count > threshold {
                    word_acc |= 1 << bit_pos;
                } else if is_even && bit_count == threshold {
                    word_acc |= tie_breaker & (1 << bit_pos);
                }
            }
            result.data[i] = word_acc;
        }
        result
    }

    fn count(&self) -> usize {
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

    pub fn permute_idx(&self, by: usize) -> Self {
        let data = std::array::from_fn(|i| self.data[(i + by) % N_WORDS]);
        Self { data }
    }

    pub fn permute_bit(&self, by: usize) -> Self {
        let shift = by % Self::DIM;
        if shift == 0 {
            return self.clone();
        }
        let mut result = Self::zero();
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
