# Hypervector

A Rust library for hyperdimensional computing (HDC).

Hyperdimensional computing is a brain-inspired paradigm where information is represented as high-dimensional vectors (hypervectors or HDVs) and processed using simple algebraic operations like addition, multiplication, and permutation. This enables fast, robust, and noise-tolerant learning for tasks like classification, symbolic reasoning, and associative memory [4].


## Features

- Binary (0/1) and bipolar (+1/-1) hypervector types (HDVs).
- Real and complex hypervector types (also known as HRRs or Holographic Reduced Representations).
- Modular Composite Representation hypervector type.
- Trait-based design for extensibility.
- Example applications: symbolic reasoning, associative memory and classification across multiple modalities.


## Examples

Runnable example problems (`cargo run --example name`):

 1. Kanerva's [Mexican Dollar](READMEmexican.md) inferencing example [1].
 2. Plate's [inferencing example](READMEplate.md) used in his HRR paper [4].
 3. [Associative memory](READMEkv.md): Benchmark retrieval accuracy vs. bundle size across HDV types and dimensions.
 4. [Activity recognition (UCI HAR)](READMEhar.md): walking, standing, sitting etc...
 5. [Spoken letter recognition (UCI Isolet)](READMEisolet.md).
 6. [Wine quality classification (UCI Wine Quality)](READMEwine.md).
 7. [Promoter Gene Sequences (UCI Molecular Biology)](READMEpromoter.md).
 8. [Iris](READMEiris.md): the classic ML dataset (flowers). 
 9. [Adult](READMEadult.md): the classic Adult dataset (Census Income). 
10. [MovieLens 100k](READMEml-100k.md): movie recommendation on the MovieLens 100k dataset. 
11. [Text language identification](READMElanguage_id.md): English, French, ... 22 languages.
12. [MNIST image classification](https://github.com/jesper-olsen/engram). Note this example is in its own repo.


## References

1. ["What We Mean When We Say 'What’s the Dollar of Mexico?'" – Pentti Kanerva, 2010](https://aaai.org/papers/02243-2243-what-we-mean-when-we-say-whats-the-dollar-of-mexico-prototypes-and-mapping-in-concept-space/)  
2. "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing" Abbas Rahimi, Pentti Kanerva, Jan M. Rabaey, 2016
3. ["Hyperdimensional Computing: An Algebra for Computing with Vectors", Pentti Kanerva, 2022](https://redwood.berkeley.edu/wp-content/uploads/2022/05/kanerva2022hdmss.pdf)
4. ["Holographic Reduced Representations", Tony Plate, IEEE Transactions on Neural Networks, February, 1995, 6(3):623-41](https://www.researchgate.net/publication/5589577_Holographic_Reduced_Representations)
5. ["Modular Composite Representation", J. Snaider S. Franklin, 2014](https://digitalcommons.memphis.edu/ccrg_papers/32/)
