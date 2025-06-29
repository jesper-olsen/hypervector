use crate::HyperVector;

impl<const DIM: usize> HyperVector for BipolarHDV<DIM> {
    fn new() -> Self {
        BipolarHDV::new()
    }

    fn distance(&self, other: &Self) -> f32 {
        1.0 - self.dot(other)
    }

    fn multiply(&self, other: &Self) -> Self {
        BipolarHDV::multiply(self, other)
    }

    fn acc(vectors: &[&Self]) -> Self {
        BipolarHDV::acc(vectors)
    }
}

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
    pub fn pmultiply(h1: &Self, order1: usize, h2: &Self, order2: usize) -> Self {
        let len = h1.data.len();
        let shift1 = order1 % len;
        let shift2 = order2 % len;
        let data =
            std::array::from_fn(|i| h2.data[(i + shift1) % len] * h2.data[(i + shift2) % len]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bipolar_mexican_dollar() {
        crate::example_mexican_dollar::<BipolarHDV<1000>>();
    }
}
