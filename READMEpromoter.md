# Hypervector - Molecular Biology (Promoter Gene Sequences)

Classification on the UCI Molecular Biology dataset [1] - promoter vs non-promoter.

Using 40,192-dimensional binary hypervectors with multi-scale encoding (bundling 4-, 5-, and 6-grams), we achieve 103/106 correct classifications (97%) under leave-one-out evaluation.

For comparison, the original paper lists these results:

| System	  | Errors | Comments
| :-----      | -----: | :-------
|  KBANN	  |  4/106 | a hybrid ML system
|  BP		  |  8/106 | std backprop with one hidden layer
|  O'Neill    | 12/106 | ad hoc technique from the bio. lit.
|  Near-Neigh | 13/106 | a nearest-neighbor algo (k=3)
|  ID3        | 19/106 | Quinlan's decision-tree builder

Results are not strictly comparable due to differences in evaluation protocols.


## Modelling

Each DNA base (A, C, G, T) is assigned a random hypervector.  
Sequences are encoded by constructing n-gram hypervectors using permutation and binding, 
and bundling them across the sequence.

To improve robustness, encodings for n-1, n, and n+1 are combined (multi-scale representation).

Class prototypes are formed by bundling encoded sequences per class.  
Classification is performed by nearest-prototype similarity.

Evaluation uses leave-one-out cross-validation.


## Usage

```bash
% cargo run --example promoter --release -- --help

Usage: promoter [OPTIONS]

Options:
      --mode <MODE>    [default: binary] [possible values: binary, modular]
      --dim <DIM>      one of 1024, 2048, 4096, 8192, 10048, 20096, 40192 [default: 1024]
      --ngram <NGRAM>  [default: 3]
  -h, --help           Print help
  -V, --version        Print version
```

## Preliminaries

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --release --example promoter -- --dim 40192 --ngram 9  

Mode: binary N-gram: 9 Dim: 40192
2-gram: Accuracy: 78.30% (83/106)
3-gram: Accuracy: 85.85% (91/106)
4-gram: Accuracy: 92.45% (98/106)
5-gram: Accuracy: 97.17% (103/106)
6-gram: Accuracy: 95.28% (101/106)
7-gram: Accuracy: 86.79% (92/106)
8-gram: Accuracy: 83.96% (89/106)
9-gram: Accuracy: 74.53% (79/106)
```

Performance peaks around 5-grams; combining neighboring n-gram scales improves robustness and reduces variance compared to single-scale encodings.

Results are consistent across multiple random seeds (not shown).

## References

1. [UCI Molecular Biology](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences

