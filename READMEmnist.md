# MNIST - Handwritten Digit Classification 

This example demonstrates that a relatively simple hyperdimensional computing pipeline, and an ensemble of perceptrons - can achieve 98.6% accuracy on MNIST without deep learning or backpropagation through multiple layers.

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
cargo run --example mnist --release -- --data-dir MNIST --augment --ensemble-size 11    

Loaded 540000 training labels
Training model 1/11
Encoding images (Dim 12800)...
Epoch 2000: 524756/540000=97.18%
Test Accuracy:  9770/10000 = 97.70%

[..snip..]

Training model 11/11
Encoding images (Dim 12800)...
Epoch 2000: 522474/540000=96.75%
Test Accuracy:  9749/10000 = 97.49%

Accuracy range: 97.1% - 98.0%

Hard-vote Ensemble Accuracy:    9860/10000 = 98.60%
Score-fusion Ensemble Accuracy:  9864/10000 = 98.64%

Score-fusion Confusion Matrix:
true\pred     0     1     2     3     4     5     6     7     8     9
       0    976     0     1     0     0     0     1     0     1     1
       1      0  1124     2     0     1     2     3     2     1     0
       2      2     4  1017     1     1     0     0     3     4     0
       3      0     0     1  1001     0     3     0     1     3     1
       4      1     0     1     1   974     0     1     0     1     3
       5      2     1     1     3     0   882     1     0     0     2
       6      3     2     0     1     3     4   944     0     1     0
       7      0     3    10     1     2     0     0  1001     2     9
       8      3     0     4     0     4     2     0     2   957     2
       9      0     3     0     2     6     2     0     3     5   988
```

## Benchmark

Dim   | Jitter | Ensemble Size | Individual Accuracy | Ensemble Accuracy | Time
-----:| :-----:| -------------:|--------------------:| -----------------:| ----:
 6400 |   no   |     5         |   97.1% - 97.4%     |     97.8%         |   3m
 6400 |  yes   |     5         |   94.5% - 97.4%     |     98.3%         |  36m
12800 |   no   |     5         |   97.4% - 97.7%     |     97.8%         |   6m
12800 |  yes   |     5         |   97.4% - 98.0%     |     98.4%         |  73m
12800 |  yes   |    11         |   97.0% - 97.8%     |     98.6%         | 131m

