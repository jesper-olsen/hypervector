# hypervector

A Rust library for hyperdimensional computing (HDC), supporting bipolar (+1/–1) binary (0/1) hypervectors and real hypervectors. Includes example applications such as logical inference and language identification.

Hyperdimensional computing (HDC) is a brain-inspired paradigm where information is represented as high-dimensional vectors—called hypervectors—and processed using simple algebraic operations like addition, multiplication, and permutation. This enables fast, robust, and noise-tolerant learning for tasks like classification, symbolic reasoning, and associative memory [4].

## Features

- Binary and bipolar hypervector types
- Trait-based design for extensibility
- Example applications: concept inference and text classification

## References

1. ["What We Mean When We Say 'What’s the Dollar of Mexico?'" – Pentti Kanerva, 2010](https://aaai.org/papers/02243-2243-what-we-mean-when-we-say-whats-the-dollar-of-mexico-prototypes-and-mapping-in-concept-space/)  
2. "Language Geometry using Random Indexing" Aditya Joshi1, Johan T. Halseth, and Pentti Kanerva, 2016
3. "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing" Abbas Rahimi, Pentti Kanerva, Jan M. Rabaey, 2016
4. ["Hyperdimensional Computing: An Algebra for Computing with Vectors", Pentti Kanerva, 2022](https://redwood.berkeley.edu/wp-content/uploads/2022/05/kanerva2022hdmss.pdf)

Run
-----

For an example of logic inferencing, see the "mexican_dollar" [1] #[test] in lib.rs - or run it with
```
% cargo run --release --bin main
```


Benchmark
---------

Here we benchmark bipolar and binary HDVs on a language identification task [2].
HDVs are used for representing letter n-grams and a profile for a language is constructed by summing over example texts in the language. For testing, a similar profile is constructed from a test sentence and compared to the reference profile (Hamming distance or cosine similarity).

There are 21 test languages (1k sentences per language) and 22 training languages (21+afr, 10k sentences each). All times are in seconds (wall clock) on a Macbook Air M1 (2020, 8 cores). 

Bipolar HDVs are represented as u8 arrays with one element per dimension. Binary HDVs are more compact
because they only use one bit per dimension; this is not only more storage efficient but also computationally significantly faster as can be seen from the table below.

```
cargo run --release --bin main_li -- --mode binary --dim 100032 --ngram 5
```

### Accuracy 
| kind    | ngram | hdv dim | hdv bits  | Accuracy  | Time      |  
| ----:   | ----: | --:     | --------: | ---------:| --------: |
| bipolar | 3     |    1024 |   8192    | 91.0%     |   263s    |
| bipolar | 3     |   10048 |  80384    | 96.7%     |  2589s    | 
| bipolar | 3     |  100032 | 800256    | 97.0%     | 29187s    | 
| binary  | 3     |    1024 |   1024    | 90.7%     |    13s    |
| binary  | 3     |   10048 |  10048    | 96.5%     |    89s    |
| binary  | 3     |  100032 | 100032    | 97.0%     |  1009s    |
| binary  | 4     |  100032 | 100032    | 98.0%     |  1031s    |
| binary  | 5     |  100032 | 100032    | 98.1%     |   953s    |
| real    | 3     |    1024 |  65536    | 96.9%     |   413s    |
| real    | 3     |   10048 | 643072    | 97.0%     |  4109s    |
| real    | 4     |   10048 | 643072    | 98.0%     |  4174s    |
| real    | 5     |   10048 | 643072    | 98.3%     |  3942s    |
| real    | 6     |   10048 | 643072    | 98.6%     |  4027s    |
