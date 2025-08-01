use crate::{Accumulator, HyperVector};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rustfft::{FftPlanner, num_complex::Complex};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::size_of;

// avoid repeated setup of FftPlanner in bind()
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealHDV<const N: usize> {
    pub data: [f64; N],
}

impl<const N: usize> HyperVector for RealHDV<N> {
    type Accumulator = RealAccumulator<N>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let stddev = 1.0 / (N as f64).sqrt();
        let normal = Normal::new(0.0, stddev).unwrap();
        let data = std::array::from_fn(|_| normal.sample(rng));
        Self { data }
    }

    fn ident() -> Self {
        Self {
            data: std::array::from_fn(|i| if i == 0 { 1.0 } else { 0.0 }),
        }
    }

    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() <= N);
        let data = std::array::from_fn(|i| {
            if i < slice.len() {
                slice[i] as f64
            } else {
                0.0
            }
        });
        Self { data }
    }

    fn distance(&self, other: &Self) -> f32 {
        self.distance_cosine_sim(other)
        //self.dot(other) as f32
    }

    fn bind(&self, other: &Self) -> Self {
        //self.bind_circular_convolution(other)
        self.bind_fft(other)
    }

    fn unbind(&self, other: &Self) -> Self {
        let oi = other.approx_inverse();
        self.bind(&oi)
    }

    fn permute(&self, by: usize) -> Self {
        self.permute(by)
    }

    fn unpermute(&self, by: usize) -> Self {
        self.unpermute(by)
    }

    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let perm_a = self.permute(pa);
        let perm_b = other.permute(pb);
        perm_a.bind(&perm_b)
    }

    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self {
        let perm_b = other.permute(pb);
        let unbound = self.unbind(&perm_b);
        unbound.unpermute(pa)
    }

    fn bundle(vectors: &[&Self]) -> Self {
        let mut sum = [0.0; N];
        for v in vectors {
            sum.iter_mut().zip(v.data.iter()).for_each(|(s, d)| *s += d);
        }
        if vectors.len() > 0 {
            sum.iter_mut()
                .for_each(|e| *e /= (vectors.len() as f64).sqrt())
        }
        RealHDV { data: sum }
    }

    fn unpack(&self) -> Vec<f32> {
        self.data.iter().map(|&e| e as f32).collect()
    }

    fn write(&self, file: &mut File) -> io::Result<()> {
        for &value in &self.data {
            file.write_all(&value.to_ne_bytes())?;
        }
        Ok(())
    }

    fn read(file: &mut File) -> io::Result<Self> {
        let mut data = [0f64; N];
        for slot in &mut data {
            let mut buf = [0u8; size_of::<f64>()];
            file.read_exact(&mut buf)?;
            *slot = f64::from_ne_bytes(buf);
        }
        Ok(Self { data })
    }
}

impl<const N: usize> RealHDV<N> {
    fn _bind_circular_convolution(&self, other: &Self) -> Self {
        // Performs circular convolution using the direct, time-domain formula:
        // result[j] = Σ (from k=0 to N-1) of other[k] * self[j-k]
        let mut result_data = [0.0; N];

        for j in 0..N {
            let mut sum = 0.0;
            for k in 0..N {
                // circular indexing - note
                // -3 % 10 == -3
                // -3.rem_euclid(10) == 7
                let idx = (j as isize - k as isize).rem_euclid(N as isize) as usize;
                sum += other.data[k] * self.data[idx];
            }
            result_data[j] = sum;
        }
        Self { data: result_data }
    }

    fn bind_fft(&self, other: &Self) -> Self {
        // time domain -> frequency domain ; multiply ; frequency domain -> time domain
        // let mut planner = FftPlanner::new();
        // let fft = planner.plan_fft_forward(N);
        // let ifft = planner.plan_fft_inverse(N);
        FFT_PLANNER.with(|planner| {
            let mut planner = planner.borrow_mut();
            let fft = planner.plan_fft_forward(N);
            let ifft = planner.plan_fft_inverse(N);

            // Convert to complex numbers
            let mut a: Vec<Complex<f64>> =
                self.data.iter().map(|&x| Complex::new(x, 0.0)).collect();
            let mut b: Vec<Complex<f64>> =
                other.data.iter().map(|&x| Complex::new(x, 0.0)).collect();
            let mut result = vec![Complex::new(0.0, 0.0); N];

            fft.process(&mut a);
            fft.process(&mut b);

            for i in 0..N {
                result[i] = a[i] * b[i];
            }

            ifft.process(&mut result);

            // Extract real parts and scale
            let scale = 1.0 / (N as f64);
            let data = std::array::from_fn(|i| result[i].re * scale);

            Self { data }
        })
    }

    fn approx_inverse(&self) -> Self {
        let data = std::array::from_fn(|i| {
            if i == 0 {
                self.data[0]
            } else {
                self.data[N - i]
            }
        });
        Self { data }
    }

    fn permute(&self, by: usize) -> Self {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[(i + by) % N];
        }
        Self { data }
    }

    fn unpermute(&self, by: usize) -> Self {
        self.permute(N - (by % N))
    }

    pub fn normalise(&mut self) {
        let norm = self
            .data
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        self.data.iter_mut().for_each(|e| *e /= norm);
    }

    fn dot(&self, other: &Self) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    }

    fn distance_cosine_sim(&self, other: &Self) -> f32 {
        let mag_a = self.data.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mag_b = other.data.iter().map(|b| b * b).sum::<f64>().sqrt();

        let cosine_similarity = self.dot(other) / (mag_a * mag_b);

        //let angle = (dot / (mag_a * mag_b)).acos() as f32;
        //angle 0 => same direction,
        //      π/2 => orthogonal,
        //      π => opposite
        (1.0 - cosine_similarity) as f32 // Convert to distance (0 = identical, 2 = opposite)
    }
}

#[derive(Debug, Clone)]
pub struct RealAccumulator<const N: usize> {
    sum: [f64; N],
    n: f64,
}

impl<const N: usize> Default for RealAccumulator<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Accumulator<RealHDV<N>> for RealAccumulator<N> {
    fn new() -> Self {
        Self {
            sum: [0.0; N],
            n: 0.0,
        }
    }

    fn add(&mut self, v: &RealHDV<N>, weight: f64) {
        for i in 0..N {
            self.sum[i] += weight * v.data[i];
        }
        self.n += weight;
    }

    fn finalize(&self) -> RealHDV<N> {
        let data: [f64; N] = std::array::from_fn(|i| self.sum[i] / (self.n as f64).sqrt());
        RealHDV { data }
    }
}

#[cfg(test)]
mod tests {
    use super::RealHDV;
    use crate::HyperVector;

    use mersenne_twister_rs::MersenneTwister64;

    #[test]
    fn bind_unbind() {
        let mut mt = MersenneTwister64::new(42);
        let a = RealHDV::<2048>::random(&mut mt);
        let b = RealHDV::<2048>::random(&mut mt);
        let bound = a.bind(&b);
        let recovered_b = bound.unbind(&a);
        let dist = b.distance(&recovered_b);
        println!("Dist: {dist}");
        assert!(dist < 0.5)
    }

    #[test]
    fn bind_random_length() {
        let mut mt = MersenneTwister64::new(42);
        let a = RealHDV::<1024>::random(&mut mt);
        let mag = a.data.iter().map(|&e| e * e).sum::<f64>().sqrt();
        println!("mag {mag}");
        // standard deviation is 1/sqrt(2N) => 0.022 for N=1024
        assert!((mag - 1.0).abs() < 0.1)
    }
}
