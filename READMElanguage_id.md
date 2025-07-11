# Hypervector - Language Identification

Benchmark
---------

Here we benchmark bipolar and binary HDVs on a language identification task [2].
The approach is to use HDVs to represent letter n-grams, and combine ngrams with superposition to form an overall profile for a given language. The distance between a training profile and a test profile is used for language classification.
Note that HRRs are not well-suited for this task, as unbinding introduces noise that makes classification unreliable.

There are 21 test languages (1k sentences per language) and 22 training languages (21+afr, 10k sentences each). All times are in seconds (wall clock) on a Macbook Air M1 (2020, 8 cores). 

Bipolar HDVs are represented as u8 arrays with one element per dimension. Binary HDVs are more compact
because they only use one bit per dimension; this is not only more storage efficient but also computationally significantly faster as can be seen from the table below.

```
cargo run --release --bin main_li -- --mode binary --dim 1024 --ngram 3
```

### Accuracy 
| Kind    | NGram | HDV dim | HDV bytes | Accuracy  | Time   |  
| ------: | ----: | ------: | --------: | -------:  | -----: |  
| bipolar | 3     |    1024 |      1024 | 91.3%     |    41s |
| binary  | 3     |    1024 |        16 | 91.5%     |    12s |
| real    | 3     |    1024 |      8192 | 91.3%     |  1810s |
| complex | 3     |    1024 |     16384 | 91.8%     |  1815s |
| bipolar | 3     |   10048 |     10048 | 96.5%     |  3030s |
| binary  | 3     |   10048 |       157 | 98.7%     |   120s |
| real    | 3     |   10048 |     80384 | 91.3%     |  1810s |
| complex | 3     |   10048 |    160768 | 91.8%     |  1815s |
| binary  | 3     |  100032 |      1563 | 97.1%     |   576s |
| binary  | 4     |  100032 |      1563 | 98.1%     |   555s |
| binary  | 5     |  100032 |      1563 | 98.2%     |   561s |
| binary  | 6     |  100032 |      1563 | 97.9%     |   576s |

View the language space - computed with [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) for a particular model:
```
% uv run tsne.py
```
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/LanguageSpace.png)


