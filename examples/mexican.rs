// Mexican Dollar example for individual types - see:
// Pentti Kanerva: What We Mean When We Say “What’s the Dollar of Mexico?”
// https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
// Calculate answer: Mexican Peso - mpe

use hypervector::types::{
    binary::Binary, binary::Bipolar, complex::ComplexHDV, modular::Modular, real::RealHDV,
};
use hypervector::{HyperVector, cleanup, gen_vars, hdv};
use mersenne_twister_rs::MersenneTwister64;

pub fn example_mexican_dollar<T: HyperVector>() {
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

hdv!(binary, HDV1, 8192);
hdv!(bipolar, HDV2, 8192);
hdv!(modular, HDV3, 10000);
hdv!(complex, HDV4, 2048);
hdv!(real, HDV5, 2048);

fn main() {
    println!("Binary");
    println!("=========");
    example_mexican_dollar::<HDV1>();
    println!("Bipolar");
    println!("=========");
    example_mexican_dollar::<HDV2>();
    println!("Modular");
    println!("=========");
    example_mexican_dollar::<HDV3>();
    println!("ComplexHDV");
    println!("=========");
    example_mexican_dollar::<HDV4>();
    println!("RealHDV");
    println!("=========");
    example_mexican_dollar::<HDV5>();
}
