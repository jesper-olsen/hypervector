pub mod binary_hdv;
pub mod bipolar_hdv;

pub trait Accumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T);
    fn finalize(self) -> T;
}

pub trait HyperVector: Sized {
    type Accumulator: Default + Accumulator<Self>;

    fn new() -> Self;
    /// Returns the identity element of the hypervector space:
    /// - Binary: all 0s (XOR identity)
    /// - Bipolar: all +1s (multiplicative identity)
    fn ident() -> Self;
    fn from_slice(slice: &[i8]) -> Self;
    fn distance(&self, other: &Self) -> f32;
    fn multiply(&self, other: &Self) -> Self;
    fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn acc(vectors: &[&Self]) -> Self;
}

pub fn example_mexican_dollar<T: HyperVector>() {
    // Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
    // https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
    // Calculate answer: Mexican Peso - mpe
    let name = T::new();
    let capital = T::new();
    let currency = T::new();

    let swe = T::new();
    let usa = T::new();
    let mex = T::new();

    let stockholm = T::new();
    let wdc = T::new();
    let cdmx = T::new();

    let usd = T::new();
    let mpe = T::new();
    let skr = T::new();

    let ustates = T::acc(&[
        &name.multiply(&usa),
        &capital.multiply(&wdc),
        &currency.multiply(&usd),
    ]);
    let _sweden = T::acc(&[
        &name.multiply(&swe),
        &capital.multiply(&stockholm),
        &currency.multiply(&skr),
    ]);
    let mexico = T::acc(&[
        &name.multiply(&mex),
        &capital.multiply(&cdmx),
        &currency.multiply(&mpe),
    ]);

    let fmu = mexico.multiply(&ustates);
    let x = fmu.multiply(&usd);

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
    use crate::{Accumulator, HyperVector, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV};

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

    #[test]
    fn test_bipolar_accumulate() {
        test_accumulate::<BipolarHDV<5>>();
    }

    #[test]
    fn test_binary_accumulate() {
        test_accumulate::<BinaryHDV<64>>();
    }

    #[test]
    fn binary_mexican_dollar() {
        crate::example_mexican_dollar::<BinaryHDV<16>>(); // 16*64 = 1024 bits
    }

    #[test]
    fn bipolar_mexican_dollar() {
        crate::example_mexican_dollar::<BipolarHDV<1024>>();
    }
}
