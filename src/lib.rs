pub mod binary_hdv;
pub mod bipolar_hdv;
pub mod complex_hdv;
pub mod encoding;
pub mod modular_hdv;
pub mod real_hdv;
pub mod trainer;

/// Generates hypervector types for specified dimensionality
/// The main reason for this macro is the constraints of const generics - the bitpacked implementations for binary and bipolar
/// hypervectors are parameterised in terms of the number of words used, not bits.
/// Examples:
///     hdv!(binary,   MyBinary,   1024);
///     hdv!(bipolar,  MyBipolar,  1024);
///     hdv!(real,     MyReal,     1024);
///     hdv!(complex,  MyComplex,  1024);
///     hdv!(modular,  MyModular,  1024);
#[macro_export]
macro_rules! hdv {
    (binary, $name:ident, $dim:expr) => {
        const _: () = assert!(
            $dim % (usize::BITS as usize) == 0,
            "DIM must be a multiple of usize::BITS"
        );
        pub type $name = BinaryHDV<{ $dim / usize::BITS as usize }>;
    };
    //(bipolar, $name:ident, $dim:expr) => {
    //    const _: () = assert!($dim % (usize::BITS as usize) == 0, "DIM must be a multiple of usize::BITS");
    //    pub type $name = BipolarHDV<{ $dim / usize::BITS as usize }>;
    //};
    (bipolar, $name:ident, $dim:expr) => {
        pub type $name = BipolarHDV<$dim>;
    };
    (real,    $name:ident, $dim:expr) => {
        pub type $name = RealHDV<$dim>;
    };
    (complex, $name:ident, $dim:expr) => {
        pub type $name = ComplexHDV<$dim>;
    };
    (modular, $name:ident, $dim:expr) => {
        pub type $name = ModularHDV<$dim>;
    };
}

use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// Generates multiple random hypervectors from a single RNG.
/// Usage: gen_vars!(rng, Type, var1, var2, var3);
#[macro_export]
macro_rules! gen_vars {
    ($rng:expr, $t:ty, $($name:ident),+) => {
        $(
            let $name = <$t>::random($rng);
        )+
    };
}

pub trait UnitAccumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T);
    fn finalize(&mut self) -> T;
    fn count(&self) -> usize;
}

pub trait Accumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T, weight: f64);
    fn finalize(&mut self) -> T;
    fn count(&self) -> f64;
}

pub trait HyperVector: Sized + Clone {
    type Accumulator: Default + Accumulator<Self>;
    type UnitAccumulator: Default + UnitAccumulator<Self>;
    const DIM: usize;

    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;
    /// Returns the identity element of the hypervector space:
    /// - Binary: all 0s (XOR identity)
    /// - Bipolar: all +1s (multiplicative identity)
    fn ident() -> Self;

    // blend two hypervectors by coping indices from other - rest from self
    fn blend(&self, other: &Self, indices: &[usize]) -> Self;

    fn distance(&self, other: &Self) -> f32; // 0..1
    fn similarity(&self, other: &Self) -> f32 {
        1.0 - self.distance(other)
    }

    fn bind(&self, other: &Self) -> Self;

    fn unbind(&self, other: &Self) -> Self;

    fn inverse(&self) -> Self;

    fn permute(&self, by: usize) -> Self;
    fn unpermute(&self, by: usize) -> Self;

    fn bundle(vectors: &[&Self]) -> Self {
        let mut acc: Self::UnitAccumulator = Self::UnitAccumulator::new();
        for v in vectors {
            acc.add(v)
        }
        acc.finalize()
    }

    fn unpack(&self) -> Vec<f32>;
    fn write(&self, file: &mut File) -> std::io::Result<()>;
    fn read(file: &mut File) -> std::io::Result<Self>;
}

pub fn save_hypervectors_to_csv<H: HyperVector>(
    filename: &str,
    vectors: &[H],
) -> Result<(), std::io::Error> {
    let mut writer = BufWriter::new(File::create(filename)?);

    for v in vectors {
        let values = v.unpack();
        for (i, val) in values.iter().enumerate() {
            write!(writer, "{val}")?;
            if i < values.len() - 1 {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }
    Ok(())
}

pub fn write_hypervectors<H: HyperVector>(vec: &[H], mut file: File) -> std::io::Result<()> {
    // Write number of HDVs
    let len = vec.len();
    file.write_all(&len.to_le_bytes())?;
    for hdv in vec {
        hdv.write(&mut file)?;
    }
    file.flush()
}

pub fn read_hypervectors<H: HyperVector>(mut file: File) -> std::io::Result<Vec<H>> {
    // Read number of HDVs
    let mut len_buf = [0u8; usize::BITS as usize / 8];
    file.read_exact(&mut len_buf)?;
    let len = usize::from_le_bytes(len_buf);
    println!("read_hypervectors: reading {len} HDVs");
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(H::read(&mut file)?);
    }
    Ok(vec)
}

pub fn cleanup<'a, T: HyperVector>(query: &T, vocab: &'a [(&str, T)]) -> &'a str {
    let mut best_label = vocab[0].0;
    let mut min_dist = query.distance(&vocab[0].1);

    for (label, v) in vocab.iter().skip(1) {
        let d = query.distance(v);
        if d < min_dist {
            min_dist = d;
            best_label = label;
        }
    }
    best_label
}

pub fn nearest<T: HyperVector>(query: &T, candidates: &[T]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut min_dist = query.distance(&candidates[0]);
    for (idx, v) in candidates.iter().enumerate().skip(1) {
        let d = query.distance(v);
        if d < min_dist {
            min_dist = d;
            best_idx = idx;
        }
    }
    (best_idx, min_dist)
}

pub fn nearest_two<T: HyperVector>(query: &T, candidates: &[T]) -> ((usize, f32), (usize, f32)) {
    assert!(candidates.len() >= 2);
    let mut first = (0, f32::MAX);
    let mut second = (0, f32::MAX);
    for (idx, v) in candidates.iter().enumerate() {
        let d = query.distance(v);
        if d < first.1 {
            second = first;
            first = (idx, d);
        } else if d < second.1 {
            second = (idx, d);
        }
    }
    (first, second)
}

pub fn example_mexican_dollar<T: HyperVector>() {
    // Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
    // https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
    // Calculate answer: Mexican Peso - mpe
    //
    let mut mt = MersenneTwister64::new(42);
    gen_vars!(
        &mut mt, T, name, capital, currency, swe, usa, mex, stockholm, wdc, cdmx, usd, mpe, skr
    );

    let ustates = T::bundle(&[&name.bind(&usa), &capital.bind(&wdc), &currency.bind(&usd)]);
    let mexico = T::bundle(&[&name.bind(&mex), &capital.bind(&cdmx), &currency.bind(&mpe)]);

    let transformation = mexico.bind(&ustates.inverse());
    let x = transformation.bind(&usd);

    let vocab = [
        ("swe", swe),
        ("usa", usa),
        ("mex", mex),
        ("stkhlm", stockholm),
        ("wdc", wdc),
        ("cdmx", cdmx),
        ("usd", usd),
        ("mpe", mpe),
        ("skr", skr),
    ];

    let ml = cleanup(&x, &vocab);
    println!("Nearest HDV is: {ml}\n\n");
    assert_eq!(ml, "mpe", "Expected mpe");
}

#[cfg(test)]
mod tests {
    use crate::{
        HyperVector, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV, complex_hdv::ComplexHDV,
        modular_hdv::ModularHDV, real_hdv::RealHDV,
    };

    fn test_permute_unpermute<T: HyperVector + std::fmt::Debug + std::cmp::PartialEq>() {
        //let mut rng = MersenneTwister64::new(42);
        let mut rng = rand::rng();
        let a = T::random(&mut rng);
        let b = a.permute(1);
        let c = b.unpermute(1);
        assert!(a != b);
        assert!(a == c)
    }

    #[test]
    fn test_binary_permute_unpermute() {
        test_permute_unpermute::<BinaryHDV<157>>();
    }

    #[test]
    fn test_bipolar_permute_unpermute() {
        test_permute_unpermute::<BipolarHDV<1024>>();
    }

    #[test]
    fn test_modular_permute_unpermute() {
        test_permute_unpermute::<ModularHDV<256>>();
    }

    fn test_bind_unbind<T: HyperVector + std::fmt::Debug + std::cmp::PartialEq>(thr: f32) {
        //let mut rng = MersenneTwister64::new(42);
        let mut rng = rand::rng();
        let a = T::random(&mut rng);
        let b = T::random(&mut rng);
        let c = a.bind(&b);
        let d = c.unbind(&b);
        let dist = a.distance(&d);
        println!("dist {dist:?} thr {thr}");
        assert!(dist <= thr);
    }

    #[test]
    fn test_modular_bind_unbind() {
        test_bind_unbind::<ModularHDV<256>>(0.0);
    }

    #[test]
    fn test_bipolar_bind_unbind() {
        test_bind_unbind::<BipolarHDV<1024>>(0.01);
    }

    #[test]
    fn test_binary_bind_unbind() {
        test_bind_unbind::<BinaryHDV<64>>(0.0);
    }

    #[test]
    fn test_real_bind_unbind() {
        test_bind_unbind::<RealHDV<1000>>(0.5);
    }

    #[test]
    fn test_complex_bind_unbind() {
        test_bind_unbind::<ComplexHDV<1000>>(0.5);
    }

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
        crate::example_mexican_dollar::<ComplexHDV<1000>>();
    }

    #[test]
    fn modular_mexican_dollar() {
        crate::example_mexican_dollar::<ModularHDV<10000>>();
    }
}
