# Hypervector - Iris 

Classification on the Iris dataset [1] (150 samples, 3 classes, 4 features in cm).

Classes:
- Iris setosa
- Iris versicolor
- Iris virginica

Using 8192-dimensional binary hypervectors we achieve 150/150 correct classifications (100%) under leave-one-out cross-validation.


## Modelling

The 4 continuous features are level-encoded into binary hypervectors and bundled into a single representation per sample.

A multi-prototype model (3 prototypes per class) is trained using a perceptron-style update rule.

```
Raw sample (4 features, cm)
[ sepal_len, sepal_wid, petal_len, petal_wid ]
                │
                ▼
        Level Encoding
   (quantize → binary HDVs)
                │
                ▼
     4 Hypervectors (8192-d)
        h1   h2   h3   h4
         │    │    │    │
         └────┴────┴────┘
                │
                ▼
           Bundling
                │
                ▼
     Sample Hypervector H
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
   Class A   Class B   Class C
 (3 protos) (3 protos) (3 protos)
        │       │        │
        └───────┼────────┘
                ▼
      Similarity Search
                │
                ▼
          Predicted Class
```

## Evaluation

Leave-one-out cross-validation (LOOCV) over all 150 samples.

Note: Iris is a small and relatively easy dataset; perfect accuracy is commonly achievable.




## Preliminaries

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --release --example iris  

Accuracy: 100.00% (150/150)
```

## References

1. [Iris](https://archive.ics.uci.edu/dataset/53/iris)

