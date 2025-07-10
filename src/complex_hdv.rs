use crate::{Accumulator, HyperVector};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rustfft::{FftPlanner, num_complex::Complex};
use std::cell::RefCell;

// avoid repeated setup of FftPlanner in bind()
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexHDV<const N: usize> {
    pub data: [Complex<f64>; N],
}

impl<const N: usize> HyperVector for ComplexHDV<N> {
    type Accumulator = ComplexAccumulator<N>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        // Set stddev so that E[‖z‖^2] = 1
        let stddev = 1.0 / (2.0 * N as f64).sqrt();
        let normal = Normal::new(0.0, stddev).unwrap();

        let data = std::array::from_fn(|_| {
            let re = normal.sample(rng);
            let im = normal.sample(rng);
            Complex::new(re, im)
        });

        Self { data }
        // let mut h = Self { data };
        // h.normalise();
        // h
    }

    // fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
    //     use rand::Rng;
    //     let data = std::array::from_fn(|_| {
    //         let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
    //         Complex::from_polar(1.0, theta)
    //     });
    //     Self { data }
    // }

    fn ident() -> Self {
        Self {
            data: std::array::from_fn(|i| {
                if i == 0 {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            }),
        }
    }

    fn from_slice(slice: &[f32]) -> Self {
        let data = std::array::from_fn(|i| {
            let re = slice.get(2 * i).copied().unwrap_or(0.0) as f64;
            let im = slice.get(2 * i + 1).copied().unwrap_or(0.0) as f64;
            Complex::new(re, im)
        });
        Self { data }
    }

    fn distance(&self, other: &Self) -> f32 {
        self.distance_cosine_sim(other) as f32
        //self.distance_dot(other) as f32
    }

    fn bind(&self, other: &Self) -> Self {
        //self.bind_circular_convolution(other);
        self.bind_fft(other)
        // let mut h = self.bind_fft(other);
        // h.normalise();
        // h
    }

    fn unbind(&self, other: &Self) -> Self {
        self.bind(&other.approx_inverse())
        //self.unbind_fft(&other)
        //
        // let mut h = self.unbind_fft(&other);
        // h.normalise();
        // h
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

    fn acc(vectors: &[&Self]) -> Self {
        let mut sum = [Complex::new(0.0, 0.0); N];
        for v in vectors {
            sum.iter_mut().zip(v.data.iter()).for_each(|(s, d)| *s += d);
        }
        if vectors.len() > 0 {
            sum.iter_mut()
                .for_each(|e| *e /= (vectors.len() as f64).sqrt())
        }
        Self { data: sum }
        // let mut h = Self { data: sum };
        // h.normalise();
        // h
    }

    fn unpack(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.data.len() * 2);
        for &c in &self.data {
            out.push(c.re as f32);
            out.push(c.im as f32);
        }
        out
    }
}

impl<const N: usize> ComplexHDV<N> {
    fn bind_circular_convolution(&self, other: &Self) -> Self {
        // Performs circular convolution using the direct, time-domain formula:
        // result[j] = Σ (from k=0 to N-1) of other[k] * self[j-k]
        let mut result_data = [Complex::new(0.0, 0.0); N];

        for j in 0..N {
            let mut sum = Complex::new(0.0, 0.0);
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

            // Clone the data (already complex)
            let mut a = self.data;
            let mut b = other.data;
            let mut result = [Complex::new(0.0, 0.0); N];

            fft.process(&mut a);
            fft.process(&mut b);

            for i in 0..N {
                result[i] = a[i] * b[i];
            }

            ifft.process(&mut result);

            // Scale by 1/N (standard for IFFT normalization)
            let scale = 1.0 / (N as f64);
            for x in &mut result {
                *x *= scale;
            }
            Self { data: result }
        })
    }

    fn unbind_fft(&self, other: &Self) -> Self {
        // same as bind_fft except line with conj()
        FFT_PLANNER.with(|planner| {
            let mut planner = planner.borrow_mut();
            let fft = planner.plan_fft_forward(N);
            let ifft = planner.plan_fft_inverse(N);

            let mut a: Vec<Complex<f64>> = self.data.iter().cloned().collect();
            let mut b: Vec<Complex<f64>> = other.data.iter().cloned().collect();

            fft.process(&mut a);
            fft.process(&mut b);

            let mut result = vec![Complex::new(0.0, 0.0); N];
            for i in 0..N {
                result[i] = a[i] * b[i].conj(); // <-- conjugate here!
            }

            ifft.process(&mut result);

            let scale = 1.0 / (N as f64);
            let data = std::array::from_fn(|i| result[i] * scale);

            Self { data }
        })
    }

    fn approx_inverse(&self) -> Self {
        let data = std::array::from_fn(|i| {
            if i == 0 {
                self.data[0].conj()
            } else {
                self.data[N - i].conj()
            }
        });
        Self { data }
    }

    fn permute(&self, by: usize) -> Self {
        let mut data = [Complex::new(0.0, 0.0); N];
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
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-12); // avoid division by zero
        self.data.iter_mut().for_each(|e| *e /= norm);
    }

    pub fn distance_dot(&self, other: &Self) -> f64 {
        // Hermitian dot product: z · w̅
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * b.conj())
            .sum::<Complex<f64>>()
            .norm_sqr()
    }

    pub fn distance_cosine_sim(&self, other: &Self) -> f64 {
        // Hermitian dot product: z · w̅
        let dot: Complex<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * b.conj())
            .sum();

        let mag_a = self.data.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let mag_b = other.data.iter().map(|b| b.norm_sqr()).sum::<f64>().sqrt();

        let cosine_similarity = dot.re / (mag_a * mag_b);
        1.0 - cosine_similarity // 0 = identical, 1 = orthogonal, 2 = opposite

        // let angle = cosine_similarity.re.clamp(-1.0, 1.0).acos();
        // angle as f32
    }
}

pub struct ComplexAccumulator<const N: usize> {
    sum: [Complex<f64>; N],
    n: usize,
}

impl<const N: usize> Default for ComplexAccumulator<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Accumulator<ComplexHDV<N>> for ComplexAccumulator<N> {
    fn new() -> Self {
        Self {
            sum: [Complex::new(0.0, 0.0); N],
            n: 0,
        }
    }

    fn add(&mut self, v: &ComplexHDV<N>) {
        for i in 0..N {
            self.sum[i] += v.data[i];
        }
        self.n += 1
    }

    fn finalize(self) -> ComplexHDV<N> {
        let data: [Complex<f64>; N] = std::array::from_fn(|i| self.sum[i] / (self.n as f64).sqrt());
        ComplexHDV { data }
    }
}

#[cfg(test)]
mod tests {
    use super::ComplexHDV;
    use crate::HyperVector;
    use mersenne_twister_rs::MersenneTwister64;

    #[test]
    fn bind_random_length() {
        let mut mt = MersenneTwister64::new(42);
        const N: usize = 1024;
        let a = ComplexHDV::<N>::random(&mut mt);
        let mag = a.data.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        println!("mag {mag}");
        // For ComplexHDV::random(), expected norm ≈ 1
        //let expected = (N as f64).sqrt();
        let expected = 1.0;
        assert!((mag - expected).abs() < 0.1);
    }
}
