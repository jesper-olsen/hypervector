use crate::{Accumulator, HyperVector};

impl<const DIM: usize> HyperVector for BipolarHDV<DIM> {
    type Accumulator = BipolarAccumulator<DIM>;

    fn new() -> Self {
        BipolarHDV::new()
    }

    fn ident() -> Self {
        BipolarHDV { data: [1; DIM] }
    }

    fn from_slice(slice: &[i8]) -> Self {
        assert_eq!(slice.len(), DIM);
        let data = std::array::from_fn(|i| if slice[i] >= 0 { 1 } else { -1 });
        BipolarHDV { data }
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

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        BipolarHDV::pmultiply(self, pa, other, pb)
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
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
        let mut data = [0; DIM];
        for i in 0..DIM {
            data[i] = match self.sum[i].cmp(&0) {
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
        BipolarHDV { data }
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

    /// sum HDVs in l self and normalise
    fn acc(l: &[&BipolarHDV<DIM>]) -> Self {
        let mut data = [0i8; DIM];
        for i in 0..DIM {
            let s: i64 = l.iter().map(|v| v.data[i] as i64).sum();
            data[i] = if s > 0 {
                1
            } else if s < 0 {
                -1
            } else {
                if rand::random() { 1 } else { -1 }
            };
        }
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
