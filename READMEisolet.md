# Hypervector - Letter Recognition

Classification on the Isolet dataset [1] - in this dataset ...

> ... 150 subjects spoke the name of each letter of the alphabet twice. Hence, we have 52 training examples from each speaker. 

An ensemble of 5 binary hypervector models achieves 93.7% test accuracy in 16 seconds wall time (Macbook Air M1). 
The size of a 2048-dim binary hypervector is 2KB - hence a model with 26 letters is 52KB and an ensemble of 5 is 260KB.

For comparison, sklearn's SVM classifier achieves [96.4%](py/isolet.py) accuracy on the same 
feature set - at the cost of a much larger model size (80x):

| Model             | Accuracy   | Size    |  Time |
| :---------------  | ---------: | ------: | -----:|
| SVM               |  96.4%     | ~20 MB  |   6s  |
| HDC               | 91.6-92.4% |  52 KB  |   2s  |
| HDC ensemble of 5 |   93.7%    | 260 KB  |   6s  |

For Isolet, the SVM model stores 4165 support vetors - 67% of the training samples - which accounts for the larger model size. 

## Modelling

The dataset comes with 617 dimensional feature vectors normalised to the interval -1 to 1.

The modelling of Isolet used here is the same as the one used for [UCI HAR](https://github.com/jesper-olsen/hypervector/blob/master/READMEhar.md)

## Usage

```bash
cargo run --release --example isolet -- --help

Usage: isolet [OPTIONS]

Options:
      --mode <MODE>                    [default: binary] [possible values: binary, bipolar, real, complex, modular]
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

Download the dataset from [1]. Unpack the zipped dataset with root dir "isolet".

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
time cargo run --release --example isolet -- --trainer perceptron --dim 2048   --mode binary --ensemble-size 5

Epoch 1000: Training Accuracy 6235/6238=99.95%
Model 1/5 - test: 91.85%  (1432/1559)

Epoch 1000: Training Accuracy 6224/6238=99.78%
Model 2/5 - test: 91.79%  (1431/1559)

Epoch 1000: Training Accuracy 6231/6238=99.89%
Model 3/5 - test: 91.40%  (1425/1559)
Ensemble of 3 - test 93.20%  (1453/1559)

Epoch 1000: Training Accuracy 6232/6238=99.90%
Model 4/5 - test: 90.83%  (1416/1559)
Ensemble of 4 - test 92.88%  (1448/1559)

Epoch 1000: Training Accuracy 6234/6238=99.94%
Model 5/5 - test: 91.28%  (1423/1559)
Ensemble of 5 - test 93.65%  (1460/1559)

Model accuracies - avg 91.43%, min 90.83%, max 91.85

24.06s user 3.73s system 469% cpu 5.920 total
```

## Experiments

The table shows runs with binary and modular hypervector models - the ensemble size is 5.
The "Bytes" column shows the byte size of an ensemble model: ensemble_size times number of letters times bytes per hypervector.
The modular hypevectors use 8 bits per dimension.

| Model             | Dim        | Individual Accuracy | Ensemble Accuracy | Time   | Bytes         |
| :---------------  | ---------: | :-----------------: | ----------------: | -----: | :------------ |
| binary            |  1024      | 88.4-91.0%          | 93.7%             |    5s  | 5 x 26 x 128  |
| binary            |  2048      | 90.8-91.9%          | 93.7%             |    6s  | 5 x 26 x 256  |
| binary            |  4096      | 92.0-91.5%          | 92.8%             |   11s  | 5 x 26 x 512  |
| binary            |  8192      | 92.1-93.0%          | 92.9%             |   21s  | 5 x 26 x 1024 |
| binary            | 16384      | 92.6-93.1%          | 93.3%             |   39s  | 5 x 26 x 2048 |
| modular           |  1024      | 92.7-93.4%          | 93.6%             |  108s  | 5 x 26 x 1024 |
| modular           |  2048      | 92.8-93.3%          | 93.7%             |  180s  | 5 x 26 x 2048 |


## References

1. [UCI Isolet dataset](https://archive.ics.uci.edu/dataset/54/isolet)

