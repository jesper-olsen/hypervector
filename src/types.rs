pub mod binary;
pub mod bipolar;
pub mod complex;
pub mod modular;
pub mod real;
pub mod traits;

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
        pub type $name = Binary<{ $dim / usize::BITS as usize }>;
    };
    //(bipolar, $name:ident, $dim:expr) => {
    //    const _: () = assert!($dim % (usize::BITS as usize) == 0, "DIM must be a multiple of usize::BITS");
    //    pub type $name = Bipolar<{ $dim / usize::BITS as usize }>;
    //};
    (bipolar, $name:ident, $dim:expr) => {
        pub type $name = Bipolar<$dim>;
    };
    (real,    $name:ident, $dim:expr) => {
        pub type $name = RealHDV<$dim>;
    };
    (complex, $name:ident, $dim:expr) => {
        pub type $name = ComplexHDV<$dim>;
    };
    (modular, $name:ident, $dim:expr) => {
        pub type $name = Modular<$dim>;
    };
}

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

#[cfg(test)]
mod tests {
    use crate::types::traits::HyperVector;
    use crate::types::{
        binary::Binary, bipolar::Bipolar, complex::ComplexHDV, modular::Modular, real::RealHDV,
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
        test_permute_unpermute::<Binary<157>>();
    }

    #[test]
    fn test_bipolar_permute_unpermute() {
        test_permute_unpermute::<Bipolar<1024>>();
    }

    #[test]
    fn test_modular_permute_unpermute() {
        test_permute_unpermute::<Modular<256>>();
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
        test_bind_unbind::<Modular<256>>(0.0);
    }

    #[test]
    fn test_bipolar_bind_unbind() {
        test_bind_unbind::<Bipolar<1024>>(0.01);
    }

    #[test]
    fn test_binary_bind_unbind() {
        test_bind_unbind::<Binary<64>>(0.0);
    }

    #[test]
    fn test_real_bind_unbind() {
        test_bind_unbind::<RealHDV<1000>>(0.5);
    }

    #[test]
    fn test_complex_bind_unbind() {
        test_bind_unbind::<ComplexHDV<1000>>(0.5);
    }
}
