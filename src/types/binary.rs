// Binary and Bipolar hypervectors.
// The two types are rolled into one, because they are almost identical.
//
// For Bipolar elements are -1 and 1.
// If we interpret bit value 0 as 1 and 1 as -1, binding (hadamard product) is xor (same as for binary):
//   0 0 => +1 +1 = +1 => 0
//   0 1 => +1 -1 = -1 => 1
//   1 0 => -1 +1 = -1 => 1
//   1 1 => -1 -1 = +1 => 0
//

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::mem::size_of;

use mersenne_twister_rs::MersenneTwister64;
use rand::{Rng, SeedableRng};

use crate::types::traits::{Accumulator, HyperVector, UnitAccumulator};

#[derive(Debug, PartialEq, Clone)]
pub struct Binary<const N: usize, const BIPOLAR: bool = false> {
    pub data: [usize; N],
}

// Aliase
pub type Bipolar<const N: usize> = Binary<N, true>;

impl<const N: usize, const BIPOLAR: bool> HyperVector for Binary<N, BIPOLAR> {
    type Accumulator = WeightedAcc<N, BIPOLAR>;
    //type Accumulator = FixPointAcc<N, BIPOLAR>;
    type UnitAccumulator = SlicedUnitAcc<N, BIPOLAR, 32>; // 1-64 bit PLANES
    const DIM: usize = N * usize::BITS as usize;

    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let data = std::array::from_fn(|_| rng.next_u64() as usize);
        Self { data }
    }

    fn ident() -> Self {
        Binary { data: [0; N] }
    }

    /// Creates a new HDV by blending `self` and `other`
    /// `indices` are bit positions where values from `other` are used.
    fn blend(&self, other: &Self, indices: &[usize]) -> Self {
        let mut masks = [0usize; N];
        for &idx in indices {
            let i = idx / (usize::BITS as usize);
            let j = idx % (usize::BITS as usize);
            masks[i] |= 1 << j;
        }

        let data = std::array::from_fn(|i| (self.data[i] & !masks[i]) | (other.data[i] & masks[i]));
        Self { data }
    }

    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other) as f32 / (N * usize::BITS as usize) as f32
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

    fn norm(&self) -> f32 {
        if BIPOLAR {
            return 1.0
        } 
        self.data.iter().map(|w| w.count_ones()).sum::<u32>() as f32 / Self::DIM as f32
    }

    fn unpack(&self) -> Vec<f32> {
        self.data
            .iter()
            .flat_map(|&word| (0..usize::BITS).map(move |j| ((word >> j) & 1) as f32))
            .collect()
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
        //     N,
        //     N as u32 * usize::BITS
        // );
        let mut data = [0usize; N];
        for slot in &mut data {
            let mut buf = [0u8; size_of::<usize>()];
            file.read_exact(&mut buf)?;
            *slot = usize::from_ne_bytes(buf);
        }
        Ok(Self { data })
    }
}

#[derive(Debug, Clone)]
pub struct WeightedAcc<const N: usize, const BIPOLAR: bool, R: Rng = MersenneTwister64> {
    votes: [[f32; usize::BITS as usize]; N], // one vote counter per bit
    //votes: Box<[[f32; usize::BITS as usize]; N]>, // one vote counter per bit
    count: f64, // total weight added
    rng: R,
}

impl<const N: usize, const BIPOLAR: bool> Default for WeightedAcc<N, BIPOLAR> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const BIPOLAR: bool> WeightedAcc<N, BIPOLAR> {
    pub const fn is_empty(&self) -> bool {
        self.count == 0.0
    }
}

impl<const N: usize, const BIPOLAR: bool, R: Rng + SeedableRng + Default>
    Accumulator<Binary<N, BIPOLAR>> for WeightedAcc<N, BIPOLAR, R>
{
    fn new() -> Self {
        Self {
            //votes: Box::new([[0.0; usize::BITS as usize]; N]),
            votes: [[0.0; usize::BITS as usize]; N],
            count: 0.0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }

    fn add(&mut self, v: &Binary<N, BIPOLAR>, weight: f64) {
        for i in 0..N {
            let word = v.data[i];
            for j in 0..usize::BITS as usize {
                // MAP: Binary 1 -> 1.0, Binary 0 -> -1.0
                let bit_signal = if ((word >> j) & 1) == 1 { 1.0 } else { -1.0 };
                self.votes[i][j] += weight as f32 * bit_signal;
            }
        }
        self.count += weight.abs();
    }

    fn finalize(&mut self) -> Binary<N, BIPOLAR> {
        let data = std::array::from_fn(|i| {
            let mut acc_word = 0usize;
            let tie_breaker: usize = self.rng.next_u64() as usize;
            for bidx in 0..usize::BITS as usize {
                // If the consensus is positive, set the bit to 1.
                // If it's 0.0, we flip a coin to avoid bias.
                if self.votes[i][bidx] > 0.0
                    || (self.votes[i][bidx] == 0.0 && (tie_breaker & (1 << bidx)) != 0)
                {
                    acc_word |= 1 << bidx;
                }
            }
            acc_word
        });
        Binary { data }
    }

    fn count(&self) -> f64 {
        self.count
    }
}

pub struct FixPointAcc<const N: usize, const BIPOLAR: bool, R: Rng = MersenneTwister64> {
    // Same as WeightedAcc, but implemented with i32 instead of f32
    //votes: Box<[[i32; usize::BITS as usize]; N]>,
    votes: [[i32; usize::BITS as usize]; N],
    count: f64,
    rng: R,
}

impl<const N: usize, const BIPOLAR: bool> Default for FixPointAcc<N, BIPOLAR> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const BIPOLAR: bool> FixPointAcc<N, BIPOLAR> {
    pub const fn is_empty(&self) -> bool {
        self.count == 0.0
    }
}

impl<const N: usize, const BIPOLAR: bool, R: Rng + SeedableRng + Default>
    Accumulator<Binary<N, BIPOLAR>> for FixPointAcc<N, BIPOLAR, R>
{
    fn new() -> Self {
        Self {
            //votes: Box::new([[0; usize::BITS as usize]; N]),
            votes: [[0; usize::BITS as usize]; N],
            count: 0.0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }

    fn add(&mut self, v: &Binary<N, BIPOLAR>, weight: f64) {
        // Scale factor of 10,000 preserves 4 decimal places
        const SCALE: f64 = 10_000.0;
        let w = (weight * SCALE) as i32;

        for i in 0..N {
            let word = v.data[i];
            for j in 0..usize::BITS as usize {
                let bit_is_set = (word >> j) & 1 == 1;
                // Accumulate the scaled integer
                if bit_is_set {
                    self.votes[i][j] += w;
                } else {
                    self.votes[i][j] -= w;
                }
            }
        }
        self.count += weight.abs();
    }

    fn finalize(&mut self) -> Binary<N, BIPOLAR> {
        let data = std::array::from_fn(|i| {
            let mut acc_word = 0usize;
            let tie_breaker: usize = self.rng.next_u64() as usize;
            for bidx in 0..usize::BITS as usize {
                // We only care about the sign of the scaled sum
                if self.votes[i][bidx] > 0
                    || (self.votes[i][bidx] == 0 && (tie_breaker & (1 << bidx)) != 0)
                {
                    acc_word |= 1 << bidx;
                }
            }
            acc_word
        });
        Binary { data }
    }

    fn count(&self) -> f64 {
        self.count
    }
}

type VoteCount = u32;

// simple, but SlicedUnitAcc is typically faster
#[derive(Debug, Clone)]
pub struct UnitAcc<const N: usize, const BIPOLAR: bool, R: Rng = MersenneTwister64> {
    votes: [[VoteCount; usize::BITS as usize]; N],
    count: usize, // total number of vectors added
    rng: R,
}

impl<const N: usize, const BIPOLAR: bool, R: Rng + SeedableRng + Default> Default
    for UnitAcc<N, BIPOLAR, R>
{
    fn default() -> Self {
        Self {
            votes: [[0; usize::BITS as usize]; N],
            count: 0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }
}

impl<const N: usize, const BIPOLAR: bool> UnitAcc<N, BIPOLAR> {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<const N: usize, const BIPOLAR: bool> UnitAccumulator<Binary<N, BIPOLAR>>
    for UnitAcc<N, BIPOLAR>
{
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, v: &Binary<N, BIPOLAR>) {
        for i in 0..N {
            let word = v.data[i];
            for j in 0..usize::BITS {
                let flag = ((word >> j) & 1) as VoteCount;
                self.votes[i][j as usize] += flag;
            }
        }
        self.count += 1;
    }

    fn finalize(&mut self) -> Binary<N, BIPOLAR> {
        let data = std::array::from_fn(|uidx| {
            let tie_breaker: usize = self.rng.next_u64() as usize;
            let mut acc_word = 0usize;
            for bidx in 0..usize::BITS as usize {
                let n1 = self.votes[uidx][bidx]; // #1s
                let n0 = (self.count - n1 as usize) as VoteCount; // #0s
                if n1 > n0 || (n1 == n0 && (tie_breaker & (1 << bidx)) != 0) {
                    acc_word |= 1 << bidx;
                }
            }
            acc_word
        });
        Binary { data }
    }

    fn count(&self) -> usize {
        self.count
    }
}

pub struct SlicedUnitAcc<
    const N: usize,
    const BIPOLAR: bool,
    const PLANES: usize,
    R: Rng = MersenneTwister64,
> {
    // Each plane is a bit-level of the parallel counters.
    // Plane 0 = Least Significant Bit, Plane PLANES-1 = Most Significant Bit.
    data: [[usize; N]; PLANES],
    count: usize,
    rng: R,
}

impl<const N: usize, const BIPOLAR: bool, const PLANES: usize, R: Rng + SeedableRng + Default>
    Default for SlicedUnitAcc<N, BIPOLAR, PLANES, R>
{
    fn default() -> Self {
        Self {
            data: [[0; N]; PLANES],
            count: 0,
            rng: R::from_rng(&mut rand::rng()),
        }
    }
}

impl<const N: usize, const BIPOLAR: bool, const PLANES: usize> UnitAccumulator<Binary<N, BIPOLAR>>
    for SlicedUnitAcc<N, BIPOLAR, PLANES>
{
    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn add(&mut self, v: &Binary<N, BIPOLAR>) {
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

    fn finalize(&mut self) -> Binary<N, BIPOLAR> {
        let threshold = (self.count / 2) as u64;
        let is_even = self.count.is_multiple_of(2);

        let data = std::array::from_fn(|i| {
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
            word_acc
        });
        Binary { data }
    }

    fn count(&self) -> usize {
        self.count
    }
}

impl<const N: usize> Binary<N, true> {
    pub fn from_slice(slice: &[i8]) -> Self {
        let dim = N * usize::BITS as usize;
        assert!(slice.len() <= dim);
        let mut hdv = Self::zero();
        for (i, e) in slice.iter().enumerate() {
            let word_idx = i / usize::BITS as usize;
            let bit_idx = i % usize::BITS as usize;
            if *e == -1 {
                hdv.data[word_idx] |= 1 << bit_idx;
            }
        }
        hdv
    }
}

impl<const N: usize> Binary<N, false> {
    pub fn from_slice(slice: &[u8]) -> Self {
        let dim = N * usize::BITS as usize;
        assert!(slice.len() <= dim);
        let mut data = [0; N];
        for (i, e) in slice.iter().enumerate() {
            let word_idx = i / usize::BITS as usize;
            let bit_idx = i % usize::BITS as usize;
            if *e != 0 {
                data[word_idx] |= 1 << bit_idx;
            }
        }
        Self { data }
    }
}

impl<const N: usize, const BIPOLAR: bool> Binary<N, BIPOLAR> {
    pub fn zero() -> Self {
        Self { data: [0usize; N] }
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&e| e == 0)
    }

    /// Returns a Vec<u8> with one entry per bit (0 or 1).
    pub fn as_u8_vec(&self) -> Vec<u8> {
        let mut bits = Vec::with_capacity(N * usize::BITS as usize);
        for word in self.data {
            for i in 0..usize::BITS {
                let bit = (word >> i) & 1;
                bits.push(bit as u8);
            }
        }
        bits
    }

    pub fn permute_idx(&self, by: usize) -> Self {
        let data = std::array::from_fn(|i| self.data[(i + by) % N]);
        Self { data }
    }

    pub fn permute_bit(&self, by: usize) -> Self {
        let shift = by % Self::DIM;
        if shift == 0 {
            return self.clone();
        }
        let mut data = [0; N];
        let word_shift = shift / usize::BITS as usize;
        let bit_shift = shift % usize::BITS as usize;

        for i in 0..N {
            let target_idx = (i + word_shift) % N;
            let next_idx = (target_idx + 1) % N;

            // Carry bits from this word to the next to simulate a global circular shift
            data[target_idx] |= self.data[i] << bit_shift;
            if bit_shift > 0 {
                data[next_idx] |= self.data[i] >> (usize::BITS as usize - bit_shift);
            }
        }
        Self { data }
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
        let data = std::array::from_fn(|i| {
            // safe because DIM is a multiple of 64 - need to mask unused bits if this is not the case
            !(self.data[i] ^ other.data[i])
        });
        Self { data }
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
    hdv_dataset: &[Binary<N>],
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
    use crate::types::binary::{Binary, Bipolar, UnitAcc, WeightedAcc};
    use crate::types::traits::{Accumulator, HyperVector, UnitAccumulator};

    #[test]
    fn test_accumulate_binary() {
        let v1 = Binary::<1>::from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
        let v2 = Binary::<1>::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let v3 = Binary::<1>::from_slice(&[1, 0, 0, 1, 0, 0, 0, 0]);
        let expected = Binary::<1>::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);

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

        let result = Binary::<1>::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_accumulate_bipolar() {
        let v1 = Bipolar::<1>::from_slice(&[-1, 1, -1, 1, 1, 1, 1, 1]);
        let v2 = Bipolar::<1>::from_slice(&[-1, 1, 1, 1, 1, 1, 1, 1]);
        let v3 = Bipolar::<1>::from_slice(&[-1, 1, 1, -1, 1, 1, 1, 1]);
        let expected = Bipolar::<1>::from_slice(&[-1, 1, 1, 1, 1, 1, 1, 1]);

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

        let result = Bipolar::<1>::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }
}
