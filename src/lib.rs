pub mod binary_hdv;
pub mod bipolar_hdv;
pub mod complex_hdv;
pub mod real_hdv;
use mersenne_twister_rs::MersenneTwister64;
use rand_core::RngCore;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

pub trait Accumulator<T: HyperVector> {
    fn new() -> Self;
    fn add(&mut self, v: &T, weight: usize);
    fn finalize(&self) -> T;
}

pub trait HyperVector: Sized {
    type Accumulator: Default + Clone + Accumulator<Self>;

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self;
    /// Returns the identity element of the hypervector space:
    /// - Binary: all 0s (XOR identity)
    /// - Bipolar: all +1s (multiplicative identity)
    fn ident() -> Self;
    fn from_slice(slice: &[f32]) -> Self;
    fn distance(&self, other: &Self) -> f32;
    fn bind(&self, other: &Self) -> Self;
    fn unbind(&self, other: &Self) -> Self;
    fn permute(&self, by: usize) -> Self;
    fn unpermute(&self, by: usize) -> Self;
    fn pbind(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn punbind(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn bundle(vectors: &[&Self]) -> Self;
    fn unpack(&self) -> Vec<f32>;
    fn write(&self, file: &mut File) -> std::io::Result<()>;
    fn read(file: &mut File) -> std::io::Result<Self>;
}

pub fn save_hypervectors_to_csv<H: HyperVector + Copy>(
    filename: &str,
    vectors: &[H],
) -> Result<(), std::io::Error> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    println!("Saving {} HDVs to {filename}", vectors.len());
    for v in vectors {
        let row: Vec<String> = v.unpack().iter().map(|x| x.to_string()).collect();
        writeln!(writer, "{}", row.join(","))?
    }
    Ok(())
}

pub fn write_hypervectors<H: HyperVector>(vec: &[H], mut file: File) -> std::io::Result<()> {
    // Write number of HDVs
    let len = vec.len();
    file.write_all(&len.to_ne_bytes())?;
    for hdv in vec {
        hdv.write(&mut file)?;
    }
    file.flush()
}

pub fn read_hypervectors<H: HyperVector>(mut file: File) -> std::io::Result<Vec<H>> {
    // Read number of HDVs
    let mut len_buf = [0u8; size_of::<usize>()];
    file.read_exact(&mut len_buf)?;
    let len = usize::from_ne_bytes(len_buf);
    println!("read_hypervectors: reading {len} HDVs");
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(H::read(&mut file)?);
    }
    Ok(vec)
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

    let ustates = T::bundle(&[&name.bind(&usa), &capital.bind(&wdc), &currency.bind(&usd)]);
    let _sweden = T::bundle(&[
        &name.bind(&swe),
        &capital.bind(&stockholm),
        &currency.bind(&skr),
    ]);

    let mut acc = T::Accumulator::default();
    acc.add(&name.bind(&mex), 1);
    acc.add(&capital.bind(&cdmx), 1);
    acc.add(&currency.bind(&mpe), 1);
    let mexico = acc.finalize();
    //let mexico = T::bundle(&[&name.bind(&mex), &capital.bind(&cdmx), &currency.bind(&mpe)]);

    let fmu = mexico.bind(&ustates);
    let x = fmu.unbind(&usd);

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
    let mut ml = vocab[0].0;
    let mut md = x.distance(&vocab[0].1);
    println!("{ml} {md:?}");
    for (label, v) in vocab.iter().skip(1) {
        let d = x.distance(v);
        println!("{label} {d:?}");
        if d < md {
            md = d;
            ml = label;
        }
    }
    println!("Nearest HDV is: {ml}\n\n");
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
        let v1 = T::from_slice(&[1.0, -1.0, 1.0, -1.0, -1.0]);
        let v2 = T::from_slice(&[1.0, -1.0, -1.0, -1.0, -1.0]);
        let v3 = T::from_slice(&[1.0, -1.0, -1.0, 1.0, -1.0]);
        let expected = T::from_slice(&[1.0, -1.0, -1.0, -1.0, -1.0]);

        acc.add(&v1, 1);
        acc.add(&v2, 1);
        acc.add(&v3, 1);
        let result = acc.finalize();
        assert_eq!(result, expected);

        let result = T::bundle(&[&v1, &v2, &v3]);
        assert_eq!(result, expected);
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
    fn test_bipolar_accumulate() {
        test_accumulate::<BipolarHDV<5>>();
    }

    #[test]
    fn test_binary_accumulate() {
        test_accumulate::<BinaryHDV<64>>();
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

    // #[test]
    // fn complex_mexican_dollar() {
    //     // fails - noisy bind-unbind
    //     crate::example_mexican_dollar::<ComplexHDV<512>>();
    // }
}
