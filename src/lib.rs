pub mod binary_hdv;
pub mod bipolar_hdv;
pub mod complex_hdv;
pub mod real_hdv;
use mersenne_twister_rs::MersenneTwister64;
use rand_core::RngCore;

pub trait Accumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T);
    fn finalize(self) -> T;
}

pub trait HyperVector: Sized {
    type Accumulator: Default + Accumulator<Self>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self;
    /// Returns the identity element of the hypervector space:
    /// - Binary: all 0s (XOR identity)
    /// - Bipolar: all +1s (multiplicative identity)
    fn ident() -> Self;
    fn from_slice(slice: &[i8]) -> Self;
    fn distance(&self, other: &Self) -> f32;
    fn bind(&self, other: &Self) -> Self;
    fn unbind(&self, other: &Self) -> Self;
    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn acc(vectors: &[&Self]) -> Self;
}

pub fn example_mexican_dollar<T: HyperVector>() {
    // Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
    // https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
    // Calculate answer: Mexican Peso - mpe
    //
    let mut mt = MersenneTwister64::new(42);
    let name = T::random(&mut mt);
    let capital = T::random(&mut mt);
    let currency = T::random(&mut mt);

    let swe = T::random(&mut mt);
    let usa = T::random(&mut mt);
    let mex = T::random(&mut mt);

    let stockholm = T::random(&mut mt);
    let wdc = T::random(&mut mt);
    let cdmx = T::random(&mut mt);

    let usd = T::random(&mut mt);
    let mpe = T::random(&mut mt);
    let skr = T::random(&mut mt);

    let ustates = T::acc(&[&name.bind(&usa), &capital.bind(&wdc), &currency.bind(&usd)]);
    let _sweden = T::acc(&[
        &name.bind(&swe),
        &capital.bind(&stockholm),
        &currency.bind(&skr),
    ]);
    let mexico = T::acc(&[&name.bind(&mex), &capital.bind(&cdmx), &currency.bind(&mpe)]);

    let fmu = mexico.bind(&ustates);
    let x = fmu.unbind(&usd);

    let vocab = [
        ("swe", swe),
        ("usa", usa),
        ("mex", mex),
        ("stockholm", stockholm),
        ("wdc", wdc),
        ("cdmx", cdmx),
        ("usd", usd),
        ("mpe", mpe),
        ("skr", skr),
    ];
    let mut ml = vocab[0].0;
    let mut md = x.distance(&vocab[0].1);
    for (label, v) in vocab.iter().skip(1) {
        let d = x.distance(v);
        println!("{label} {d:?}");
        if d < md {
            md = d;
            ml = label;
        }
    }
    println!("Min is: {ml}\n\n");
    assert_eq!(ml, "mpe", "Expected mpe");
}

#[cfg(test)]
mod tests {
    use crate::{
        Accumulator, HyperVector, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV,
        complex_hdv::ComplexHDV, real_hdv::RealHDV,
    };

    fn test_accumulate<T: HyperVector + std::fmt::Debug + std::cmp::PartialEq>()
    where
        T::Accumulator: Accumulator<T> + Default,
    {
        let mut acc = T::Accumulator::default();
        let v1 = T::from_slice(&[1, -1, 1, -1, -1]);
        let v2 = T::from_slice(&[1, -1, -1, -1, -1]);
        let v3 = T::from_slice(&[1, -1, -1, 1, -1]);
        let expected = T::from_slice(&[1, -1, -1, -1, -1]);

        acc.add(&v1);
        acc.add(&v2);
        acc.add(&v3);
        let result = acc.finalize();
        assert_eq!(result, expected);

        let result = T::acc(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
    }

    fn test_bind_unbind<T: HyperVector + std::fmt::Debug + std::cmp::PartialEq>() {
        //let mut rng = MersenneTwister64::new(42);
        let mut rng = rand::rng();
        let a = T::random(&mut rng);
        let b = T::random(&mut rng);
        let c = a.bind(&b);
        let d = c.unbind(&b);
        let dist = a.distance(&d);
        println!("dist {dist:?}");
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_bipolar_accumulate() {
        test_accumulate::<BipolarHDV<5>>();
    }

    #[test]
    fn test_binary_accumulate() {
        test_accumulate::<BinaryHDV<64>>();
    }

    #[test]
    fn test_bipolar_bind_unbind() {
        test_bind_unbind::<BipolarHDV<1024>>();
    }

    #[test]
    fn test_binary_bind_unbind() {
        test_bind_unbind::<BinaryHDV<64>>();
    }

    // #[test]
    // fn test_real_bind_unbind() {
    //     test_bind_unbind::<RealHDV<1000>>();
    // }

    #[test]
    fn binary_mexican_dollar() {
        crate::example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024 bits
    }

    #[test]
    fn bipolar_mexican_dollar() {
        crate::example_mexican_dollar::<BipolarHDV<1024>>();
    }

    #[test]
    fn real_mexican_dollar() {
        crate::example_mexican_dollar::<RealHDV<2048>>();
    }

    #[test]
    fn complex_mexican_dollar() {
        crate::example_mexican_dollar::<ComplexHDV<2048>>();
    }
}
