pub mod binary_hdv;
pub mod bipolar_hdv;

pub trait HyperVector: Sized {
    fn new() -> Self;
    fn distance(&self, other: &Self) -> f32;
    fn multiply(&self, other: &Self) -> Self;
    fn pmultiply(&self, pa: usize, other: &Self, pb: usize) -> Self;
    fn acc(vectors: &[&Self]) -> Self;
}

pub fn example_mexican_dollar<T: HyperVector>() {
    //pub fn example_mexican_dollar<const DIM: usize>() {
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
    println!("Min is: {ml}");
    assert_eq!(ml, "mpe", "Expected mpe");
}
