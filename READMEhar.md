# Hypervector - Human Activity Recognition Using Smartphones

Classification on the UCI HAR dataset [1] - the dataset has sensor data from a:

> ... group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz.

An ensemble of 9 binary hypervector models achieves 93.7% test accuracy in 16 seconds wall time (Macbook Air M1). 
The size of a 8192-dim binary hypervector is 1KB - hence a model with 6 activities is 6KB and an ensemble of 5 is 30KB.

For comparison, sklearn's SVM and MLP classifiers achieve [~95%](https://github.com/jesper-olsen/UCI-Human-Activity-Recognition) accuracy on the same 
feature set - at the cost of much larger model sizes:

| Model             | Accuracy   | Size    |
| :---------------  | ---------: | ------: |
| SVM               |  ~95%      | ~10 MB  |
| MLP               |  ~95%      | ~1.8 MB |
| HDC               | 91.6-92.4% |   6KB   |
| HDC ensemble of 5 |   93.7%    | 30 KB   |

## Modelling
------------

The dataset comes with raw sensor data as well as 561 dimensional feature vectors extracted from same; Features are normalised to the interval -1 to 1.

Four different training schemes are implemented:
* Perceptron - one hypervector prototype per activity
* Multi Perceptron - N prototypes per activity.
* Passive-Aggressive - with variants Pa, PaI & PaII [3]. One hypervector prototype per activity.
* LVQ2.1 [4] - N prototypes per activity.

Encoding: Each sample is encoded by generating one random hypervector per feature and bundling them weighted by the feature values. The result is one hypervector per sample.

## Usage
--------

```bash
cargo run --release --bin har -- --help
Usage: har [OPTIONS]

Options:
      --mode <MODE>                    [default: binary] [possible values: binary, bipolar, real, complex, modular]
      --dim <DIM>                      One of 1024, 2048, 8192, 16384 [default: 8192]
      --trainer <TRAINER>              [default: Perceptron] [possible values: perceptron, pa, pai, paii, multi, lvq]
      --prototypes <PROTOTYPES>        number of prototypes per class [default: 1]
      --window <WINDOW>                lvq window [default: 0.25]
      --epochs <EPOCHS>                [default: 1000]
      --ensemble-size <ENSEMBLE_SIZE>  [default: 9]
  -h, --help                           Print help
  -V, --version                        Print version
```

## Preliminaries
----------------
Download the dataset from [1]. Unpack the zipped dataset with root dir "UCI HAR Dataset".

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run
--------------

```bash
% time cargo run --release --bin har -- --trainer perceptron --dim  8192   --mode binary --ensemble-size 5

Epoch 1000: Training Accuracy 7185/7352=97.73%
Model 1/5 - test: 92.37%  (2722/2947)

Epoch 1000: Training Accuracy 7271/7352=98.90%
Model 2/5 - test: 91.58%  (2699/2947)

Epoch 1000: Training Accuracy 7267/7352=98.84%
Model 3/5 - test: 91.72%  (2703/2947)
Ensemble of 3 - test 93.28%  (2749/2947)

Epoch 1000: Training Accuracy 7259/7352=98.74%
Model 4/5 - test: 92.30%  (2720/2947)
Ensemble of 4 - test 93.11%  (2744/2947)

Epoch 1000: Training Accuracy 7270/7352=98.88%
Model 5/5 - test: 92.26%  (2719/2947)
Ensemble of 5 - test 93.69%  (2761/2947)

82.23s user 3.49s system 531% cpu 16.129 total
```

## Experiments
--------------

The table shows runs with binary and modular hypervector models - ensemble size is 5.

| Model             | Dim        | Individual Accuracy | Ensemble Accuracy | Time   | Bytes    |
| :---------------  | ---------: | :-----------------: | ----------------: | -----: | -------: |
| binary            |  1024      | 85.4-89.1%          | 92.3%             |    6s  | 5 x 128  |
| binary            |  2048      | 87.6-91.5%          | 93.4%             |    6s  | 5 x 256  |
| binary            |  4096      | 89.7-92.0%          | 93.4%             |    9s  | 5 x 512  |
| binary            |  8192      | 91.6-93.3%          | 93.7%             |   16s  | 5 x 1024 |
| binary            | 16384      | 91.7-92.1%          | 92.8%             |   28s  | 5 x 2048 |
| modular           |  1024      | 91.9-92.7%          | 93.6%             |   26s  | 5 x 1024 |
| modular           |  2048      | 92.6-93.2%          | 93.7%             |   51s  | 5 x 2048 |


Observations
------------

The Perceptron trainer achieves the best accuracy-to-speed ratio. Multi-prototype and LVQ2.1 schemes did not improve on it for this dataset and encoding, likely because the single-prototype perceptron already captures the class structure well in the random projection space.

Encoding: The weighted random projection preserves the linear separability of the pre-engineered 561 features, which is why the simple perceptron is competitive with SVM.

Modular hypervectors (8 bits per component) achieve similar accuracy to binary at equivalent memory footprint, with better accuracy at lower dimensions — at the cost of slower training due to trigonometric accumulation.


References
----------
1. [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. ["A Public Domain Dataset for Human Activity Recognition using Smartphones", D. Anguita, A. Ghio, L. Oneto, X. Parra, Jorge Luis Reyes-Ortiz, The European Symposium on Artificial Neural Networks, 2013](https://www.semanticscholar.org/paper/A-Public-Domain-Dataset-for-Human-Activity-using-Anguita-Ghio/83de43bc849ad3d9579ccf540e6fe566ef90a58e)
3. ["Online Passive-Aggressive Algorithms", Koby Crammer et al, Journal of Machine Learning Research 7 (2006) 551–585, 2006](https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
4. T. Kohonen, "Improved versions of learning vector quantization", IJCNN 1990.

