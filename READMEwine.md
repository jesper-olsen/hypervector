# Hypervector - Wine Quality

Classification on the winequality dataset [1] - in this dataset there are 11 feature (acidity, pH, ...) and a quality rating.

In the following, the ratings where merged into Bad (<=5) and Good (>5).

An ensemble of 5 modular hypervector models achieves 79.8% test accuracy in 4 seconds wall time (Macbook Air M1). 

HDV models achieve accuracy competitive with Gradient Boosting, while remaining far smaller than Random Forest. 
The primary advantages over sklearn are online learnability and suitability for resource-constrained hardware.

Model | Accuracy | Size 
:---                          | ----: | -------:
Gradient Boosting (sklearn)   | 79.5% |  448.0 KB
Random Forest (sklearn)       | 79.3% | 4242.0 KB
HDV ensemble/modular (Rust)   | 79.8% |  660.0 KB
Logistic Regression (sklearn) | 73.5% |    0.8 KB
SGD (sklearn)                 | 69.5% |    1.0 KB


The size of a 2048-dim binary hypervector is 2KB - hence a model with 2 classes (good & bad) and the 11 features encoded with a level encoder
with 64 levels is (64 + 2)x2048 = is 132KB and an ensemble of 5 is 660KB.

## Modelling

The 11 scalar features were level encoded (64 levels) and bundled over the two classes.

## Usage

```bash
cargo run --release --example wine -- --help

Usage: wine [OPTIONS]

Options:
      --mode <MODE>                    [default: binary] [possible values: binary, bipolar, real, complex, modular]
      --colour <COLOUR>                Wine type [default: red] [possible values: red, white]
      --dim <DIM>                      One of 1024, 2048, 4096, 8192, 16384 [default: 8192]
      --trainer <TRAINER>              [default: perceptron] [possible values: perceptron, pa, pai, paii, multi, lvq]
      --prototypes <PROTOTYPES>        number of prototypes per class [default: 1]
      --window <WINDOW>                lvq window [default: 0.25]
      --epochs <EPOCHS>                [default: 1000]
      --ensemble-size <ENSEMBLE_SIZE>  [default: 9]
  -h, --help                           Print help
  -V, --version                        Print version
```

## Preliminaries

The dataset mapped from 10 to 2 quality ratings (good & bad) is in [RED](DATA/WINEQUALITY_RED) and
[WHITE](DATA/WINEQUALITY_WHITE).

Alternatively you can download it from [1] and run [py/winequality.py](py/winequality.py) to split and map it from the source.

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
time cargo run --release --example wine -- --dim 2048 --trainer multi  --prototypes 8 --ensemble-size 5 --mode modular
Epoch 203: Training Accuracy 1199/1199=100.00%
Model 1/5 - test: 77.50%  (310/400)

Epoch 120: Training Accuracy 1199/1199=100.00%
Model 2/5 - test: 78.00%  (312/400)

Epoch 167: Training Accuracy 1199/1199=100.00%
Model 3/5 - test: 78.00%  (312/400)
Ensemble of 3 - test 80.00%  (320/400)

Epoch 126: Training Accuracy 1199/1199=100.00%
Model 4/5 - test: 76.25%  (305/400)
Ensemble of 4 - test 79.25%  (317/400)

Epoch 174: Training Accuracy 1199/1199=100.00%
Model 5/5 - test: 79.50%  (318/400)
Ensemble of 5 - test 79.75%  (319/400)

Model accuracies - avg 77.85%, min 76.25%, max 79.50
cargo run --release --example wine -- --dim 2048 --trainer multi --prototypes 8    
17.88s user 0.84s system 441% cpu 4.242 total
```

## Experiments

The table shows runs with binary and modular hypervector models - the ensemble size is 5.
The "Bytes" column shows the byte size of an ensemble model. The modular hypevectors use 8 bits per dimension.
Six different trainers are used:

1. Perceptron trainer - one hypervector prototype per class trained with the perceptron rule. 
2. Multi - n prototypes per class trained with the perceptron rule.
3. Pa - Passive-Agressive base case [2]. 
4. Pai - Passive-Agressive variant i [2].
5. Paii - Passive-Agressive variant ii [2].
6. LVQ2.1 - Learning Vector Quantisation [3].


| Model             | Trainer    | Prototypes | Dim        | Individual Accuracy | Ensemble Accuracy | Time   |     Bytes           |
| :---------------  | :--------- | ----------:| ---------: | :-----------------: | ----------------: | -----: |     :-------------- |
| binary            | Perceptron |         1  |  1024      | 61.8-73.0%          | 71.8%             |    2s  | 5 x (2 + 64) x 128  |
| binary            | Perceptron |         1  |  2048      | 70.8-74.5%          | 74.7%             |    3s  | 5 x (2 + 64) x 256  |
| binary            | Perceptron |         1  |  4096      | 69.3-73.5%          | 75.3%             |    3s  | 5 x (2 + 64) x 512  |
| binary            | Perceptron |         1  |  8192      | 66.5-74.8%          | 75.3%             |    4s  | 5 x (2 + 64) x 1024 |
| binary            | Perceptron |         1  | 16384      | 70.0-76.5%          | 77.3%             |    7s  | 5 x (2 + 64) x 2048 |
| binary            | Multi      |         2  |  1024      | 67.8-71.0%          | 73.0%             |    3s  | 5 x (2 + 64) x 128  |
| binary            | Multi      |         2  |  2048      | 72.0-76.3%          | 78.0%             |    3s  | 5 x (2 + 64) x 128  |
| binary            | Multi      |         2  |  4096      | 70.8-76.5%          | 77.8%             |    3s  | 5 x (2 + 64) x 128  |
| binary            | Multi      |         3  |  4096      | 74.0-77.0%          | 78.0%             |    3s  | 5 x (2 + 64) x 128  |
| binary            | Perceptron |         1  |  1024      | 61.8-73.0%          | 71.8%             |    2s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  |  1024      | 64.0-75.3%          | 74.5%             |   14s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  |  1024      | 63.8-75.3%          | 78.5%             |   14s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  |  2048      | 70.5-74.7%          | 76.3%             |   22s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  |  4096      | 73.5-77.0%          | 77.3%             |   68s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  |  4096      | 73.8-77.3%          | 77.3%             |   79s  | 5 x (2 + 64) x 128  |
| binary            | Pa         |         1  | 16384      | 76.8-78.0%          | 79.3%             |  173s  | 5 x (2 + 64) x 128  |
| binary            | Pai        |         1  |  1024      | 63.5-73.5%          | 74.0%             |   14s  | 5 x (2 + 64) x 128  |
| binary            | Pai        |         1  |  2048      | 66.8-74.8%          | 74.0%             |   22s  | 5 x (2 + 64) x 128  |
| binary            | Pai        |         1  |  4096      | 72.0-76.3%          | 76.0%             |   56s  | 5 x (2 + 64) x 128  |
| binary            | Paii       |         1  |  1024      | 68.3-74.3%          | 76.8%             |   14s  | 5 x (2 + 64) x 128  |
| binary            | Paii       |         1  |  1024      | 70.0-73.5%          | 77.3%             |   14s  | 5 x (2 + 64) x 128  |
| binary            | Paii       |         1  |  2048      | 71.0-77.3%          | 78.3%             |   23s  | 5 x (2 + 64) x 128  |
| binary            | Paii       |         1  |  4096      | 74.3-79.5%          | 78.0%             |   82s  | 5 x (2 + 64) x 128  |
| binary            | Paii       |         1  | 16384      | 77.3-79.3%          | 79.0%             |  197s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         1  |  1024      | 71.0-74.3%          | 73.0%             |    5s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         2  |  1024      | 72.3-74.5%          | 74.3%             |    5s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         3  |  1024      | 72.8-75.3%          | 75.3%             |    4s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         4  |  1024      | 74.0-75.8%          | 75.8%             |    4s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         5  |  1024      | 75.5-78.0%          | 76.3%             |    4s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         6  |  1024      | 74.2-77.3%          | 76.8%             |    3s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         7  |  1024      | 73.8-78.8%          | 77.3%             |    4s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         8  |  1024      | 74.8-78.8%          | 77.8%             |    4s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         2  |  2048      | 72.0-74.3%          | 74.0%             |    5s  | 5 x (2 + 64) x 128  |
| binary            | lvq        |         2  |  4096      | 72.3-74.8%          | 73.5%             |    5s  | 5 x (2 + 64) x 128  |
| modular           | Perceptron |         1  |  1024      | 67.5-75.0%          | 74.0%             |    5s  | 5 x (2 + 64) x 1024 |
| modular           | Perceptron |         1  |  2048      | 67.8-72.8%          | 75.0%             |    7s  | 5 x (2 + 64) x 2048 |
| modular           | Multi      |         8  |  2048      | 76.3-79.5%          | **79.8%**         |    4s  | 5 x (2 + 64) x 2048 |
| modular           | pa         |         1  |  2048      | 73.8-77.0%          | 76.8%             |  164s  | 5 x (2 + 64) x 2048 |


## References

1. [Wine Quality dataset](https://uci-ics-mlr-prod.aws.uci.edu/dataset/186/wine%2Bquality)
2. ["Online Passive-Aggressive Algorithms", Koby Crammer et al, Journal of Machine Learning Research 7 (2006) 551–585, 2006](https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
3. T. Kohonen, "Improved versions of learning vector quantization", IJCNN 1990.

