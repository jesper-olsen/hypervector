# MNIST Classification via Vector Symbolic Architectures

This example demonstrates handwritten digit classification using hyperdimensional computing (HDC). 
Rather than relying on a deep convolutional neural network for feature extraction, 
it encodes raw $28 \times 28$ pixel images directly into high-dimensional binary vectors 
and trains an ensemble of linear perceptrons.

## Architecture

The classification pipeline consists of three core components:

1.  **Image Encoder:** Maps raw pixels into a unified hypervector. It combines a "pixel bag-of-words" (binding positional vectors to scaled intensity vectors) with structural edge features extracted via a $3 \times 3$ Sobel operator (capturing horizontal, vertical, and diagonal gradients).
2.  **Perceptron Classifier:** Iteratively updates class hypervectors by adding or subtracting the encoded image vectors based on prediction errors during training.
3.  **Ensemble Voting:** Trains multiple independent classifiers from different random initializations. The final prediction is a simple majority vote across the ensemble, which effectively smooths out the variance of any single model's hyperplane margins.

## MNIST Dataset

Download the [MNIST](https://github.com/jesper-olsen/mnist-rs) dataset.


## Usage

```
cargo run --example mnist -- --help
A demo application to showcase mnist handwritten digit classification

Usage: mnist [OPTIONS]

Options:
  -d, --data-dir <DATA_DIR>            Path to the directory containing the MNIST dataset files [default: MNIST]
  -e, --ensemble-size <ENSEMBLE_SIZE>  Number of individual classifiers to train [default: 5]
      --augment                        Augment training images by jittering
  -h, --help                           Print help
  -V, --version                        Print version
```

## Example Run

The following commandline trains an ensemble of 5 with the dataset stored in the default location (`./MNIST`).

The `--augment` options adds jitter to the training set - augments the set with images that have been shifted 
+/- 1 pixel vertically and or horizontally. This increases the number of training images from 60k to 540k.


```
cargo run --example mnist --release -- --augment
Read 540000 training labels
Training model 1
Encoding images (Dim 12800)...
Epoch 2000: 525546/540000=97.32%
Test Accuracy:  9746/10000 = 97.46%

[..snip..]

Training model 5
Encoding images (Dim 12800)...
Epoch 2000: 527744/540000=97.73%
Test Accuracy:  9796/10000 = 97.96%

Accuracy range: 97.4% - 98.0%

Ensemble Accuracy:  9836/10000 = 98.36%

Ensemble Confusion Matrix:
true\pred     0     1     2     3     4     5     6     7     8     9
       0    976     0     0     1     0     0     1     0     2     0
       1      0  1123     3     0     1     1     5     1     1     0
       2      2     4  1014     2     1     0     0     4     5     0
       3      0     0     1  1000     0     3     0     1     4     1
       4      1     0     2     1   972     0     1     0     1     4
       5      2     0     1     5     0   881     2     0     1     0
       6      3     2     0     1     2     5   945     0     0     0
       7      0     2     9     2     2     0     0  1001     2    10
       8      7     0     5     5     6     3     1     3   941     3
       9      0     3     1     2     4     2     0     7     7   983
```

## Benchmark

Dim   | Jitter | Ensemble Size | Individual Accuracy | Ensemble Accuracy | Time
-----:| :-----:| -------------:|--------------------:| -----------------:| ----:
 6400 |   no   |     5         |   97.1% - 97.4%     |     97.8%         |  3m
 6400 |  yes   |     5         |   94.5% - 97.4%     |     98.3%         | 36m
12800 |   no   |     5         |   97.4% - 97.7%     |     97.8%         |  6m
12800 |  yes   |     5         |   97.4% - 98.0%     |     98.4%         | 73m

