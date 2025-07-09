# Hypervector

A Rust library for hyperdimensional computing (HDC).

Hyperdimensional computing is a brain-inspired paradigm where information is represented as high-dimensional vectors (hypervectors or HDVs) and processed using simple algebraic operations like addition, multiplication, and permutation. This enables fast, robust, and noise-tolerant learning for tasks like classification, symbolic reasoning, and associative memory [4].

## Features

- Binary (0/1) and bipolar (+1/-1) hypervector types (HDVs).
- Real and complex hypervector types (also known as HRRs or Holographic Reduced Representations).
- Trait-based design for extensibility
- Example applications: concept inference and text classification

## References

1. ["What We Mean When We Say 'What’s the Dollar of Mexico?'" – Pentti Kanerva, 2010](https://aaai.org/papers/02243-2243-what-we-mean-when-we-say-whats-the-dollar-of-mexico-prototypes-and-mapping-in-concept-space/)  
2. "Language Geometry using Random Indexing" Aditya Joshi1, Johan T. Halseth, and Pentti Kanerva, 2016
3. "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing" Abbas Rahimi, Pentti Kanerva, Jan M. Rabaey, 2016
4. ["Hyperdimensional Computing: An Algebra for Computing with Vectors", Pentti Kanerva, 2022](https://redwood.berkeley.edu/wp-content/uploads/2022/05/kanerva2022hdmss.pdf)
5. ["Holographic Reduced Representations", Tony Plate, IEEE Transactions on Neural Networks, February, 1995, 6(3):623-41](https://www.researchgate.net/publication/5589577_Holographic_Reduced_Representations)

Run
-----

There are three examples that can all be run from the commandline:

* Kanerva's Mexican Dollar [1] inferencing example - run it with:
  ```
  % cargo run --release --bin main_mexican
  ```
* Plate's [inferencing example](READMEplate.md) used in his HRR paper [5].
* [Text language identification](READMElanguage_id.md) on a 22 language corpus [2].

