use crate::{Accumulator, HyperVector};

impl<const DIM: usize> HyperVector for BipolarHDV<DIM> {
    type Accumulator = BipolarAccumulator<DIM>;

    fn new() -> Self {
        BipolarHDV::new()
    }

    fn from_slice(slice: &[i8]) -> Self {
        assert_eq!(slice.len(), DIM);
        let mut hdv = Self::zero();
        for i in 0..DIM {
            hdv.data[i] = if slice[i] >= 0 { 1 } else { -1 };
        }
        hdv
    }

    fn distance(&self, other: &Self) -> f32 {
        1.0 - self.dot(other)
    }

    fn multiply(&self, other: &Self) -> Self {
        BipolarHDV::multiply(self, other)
    }

    fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BipolarHDV::pmultiply(self, pa, other, pb)
    }

    fn acc(vectors: &[&Self]) -> Self {
        BipolarHDV::acc(vectors)
    }
}

pub struct BipolarAccumulator<const DIM: usize> {
    sum: [i64; DIM],
}

impl<const DIM: usize> Default for BipolarAccumulator<DIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const DIM: usize> Accumulator<BipolarHDV<DIM>> for BipolarAccumulator<DIM> {
    //impl<const DIM: usize> BipolarAccumulator<DIM> {
    fn new() -> Self {
        Self { sum: [0; DIM] }
    }

    fn add(&mut self, v: &BipolarHDV<DIM>) {
        for i in 0..DIM {
            self.sum[i] += v.data[i] as i64;
        }
    }

    fn finalize(self) -> BipolarHDV<DIM> {
        let mut result = BipolarHDV::<DIM>::zero();
        for i in 0..DIM {
            result.data[i] = match self.sum[i].cmp(&0) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => {
                    if rand::random() {
                        1
                    } else {
                        -1
                    }
                }
            };
        }
        result
    }
}

#[derive(Debug, PartialEq)]
pub struct BipolarHDV<const DIM: usize> {
    data: [i8; DIM], // +1 or -1
}

impl<const DIM: usize> BipolarHDV<DIM> {
    fn new() -> Self {
        let data = std::array::from_fn(|_| if rand::random::<bool>() { 1 } else { -1 });
        Self { data }
    }

    fn zero() -> Self {
        Self { data: [0i8; DIM] }
    }

    /// sum HDVs in l self and normalise
    fn acc(l: &[&BipolarHDV<DIM>]) -> Self {
        let mut a = BipolarHDV::<DIM>::zero();
        for i in 0..DIM {
            let mut s = 0i64;
            for v in l {
                s += v.data[i] as i64;
            }
            a.data[i] = if s > 0 {
                1
            } else if s < 0 {
                -1
            } else {
                if rand::random() { 1 } else { -1 }
            };
        }
        a
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

    pub fn dot(&self, other: &Self) -> f32 {
        let dot: i32 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a * b) as i32)
            .sum();
        dot as f32 / self.data.len() as f32
    }
}
