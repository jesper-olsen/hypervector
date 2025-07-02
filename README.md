# hypervector

A Rust library for hyperdimensional computing (HDC), supporting bipolar (+1/–1) and binary (0/1) hypervectors. Includes applications such as logical inference and language identification.

> “Hyperdimensional computing extends the traditional (von Neumann) model of computing with numbers...”
> — [Kanerva, 2022](https://redwood.berkeley.edu/wp-content/uploads/2022/05/kanerva2022hdmss.pdf)

## Features

- Binary and bipolar hypervector types
- Trait-based design for extensibility
- Example applications: concept inference and text classification

## References

1. ["What We Mean When We Say 'What’s the Dollar of Mexico?'" – Pentti Kanerva, 2010](https://aaai.org/papers/02243-2243-what-we-mean-when-we-say-whats-the-dollar-of-mexico-prototypes-and-mapping-in-concept-space/)  
2. "Language Geometry using Random Indexing" – Aditya Joshi et al., 2016  
3. "A Robust and Energy-Efficient Classifier..." – Abbas Rahimi et al., 2016  
4. ["Hyperdimensional Computing: An Algebra..." – Pentti Kanerva, 2022](https://redwood.berkeley.edu/wp-content/uploads/2022/05/kanerva2022hdmss.pdf)

Run
-----

For an example of logic inferencing, see the "mexican_dollar" [1] #[test] in lib.rs - or run it with
```
% cargo run --release --bin main
```


Benchmark
---------

Here we benchmark bipolar and binary HDVs on a language identification task [2].
HDVs are used for representing letter n-grams and a profile for a language is constructed by summing over a
example texts in the language. For testing, a similar profile is constructed from a test sentence and compared to the reference profile (Hamming distance or cosine similarity).

There are 21 test languages (1k sentences per language) and 22 training languages (21+afr, 10k sentences each). All times are in seconds (wall clock) on a Macbook Air M1 (2020, 8 cores). 

Bipolar HDVs are represented as u8 arrays with one element per dimension. Binary HDVs are more compact
because they only use one bit per dimension; this is not only more storage efficient but also computationally faster as can be seen from the table below.

```
cargo run --release --bin main_li -- --mode binary --dim 100032 --ngram 5
```

### Accuracy 
| kind    | ngram | hdv bits| Accuracy    | Time       |  
| ----:   | ----: | --:     | ---------:  | ----------:| 
| bipolar | 3     |    1024 | 91.0%       |   263s     |
| bipolar | 3     |   10048 | 96.5%       |  2615s     | 
| binary  | 3     |    1024 | 90.7%       |    13s     |
| binary  | 3     |   10048 | 96.5%       |    89s     |
| binary  | 3     |  100032 | 97.0%       |  1009s     |
| binary  | 4     |  100032 | 98.0%       |  1031s     |
| binary  | 5     |  100032 | 98.1%       |   953s     |

