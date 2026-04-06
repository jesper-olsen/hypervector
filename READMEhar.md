# Hypervector - Human Activity Recognition Using Smartphones

Classification on the UCI HAR dataset [1]:

> The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz.

See [here](https://github.com/jesper-olsen/UCI-Human-Activity-Recognition) for a jupyter notebook which explores the datasets and evaluates a number of
standard classifiers from the python ecosystem.

Modelling
---------

Four different training schemes are implemented:
* Perceptron - one hypervector prototype per activity
* Multi Perceptron - N prototypes per activity.
* Passive Agressive - with variants Pa, PaI & PaII [3]. One hypervector prototype per activity.
* LVQ2.1 [4] - N prototypes per activity.

The dataset comes with raw sensor data as well as 561 dimensional feature vectors extracted from same. Features are normalised to the interval -1 to 1.

Encoding: One hypervector per feature is gerated and features bundled over each of the 6 activities.

Usage
---------

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

Preliminaries
-------------
Download the dataset from [1]. Unpack the zipped dataset with root dir "UCI HAR Dataset".

[Install Rust](https://rust-lang.org/tools/install/).


Experiments
------------

```bash
time cargo run --release --bin har -- --trainer perceptron --dim  8192   --mode binary

Epoch 1000: Training Accuracy 7273/7352=98.93%
Model 1/9 - test: 92.03%  (2712/2947)

Epoch 1000: Training Accuracy 7221/7352=98.22%
Model 2/9 - test: 91.38%  (2693/2947)

Epoch 1000: Training Accuracy 7236/7352=98.42%
Model 3/9 - test: 91.82%  (2706/2947)
Ensemble of 3 - test 93.32%  (2750/2947)

Epoch 1000: Training Accuracy 7154/7352=97.31%
Model 4/9 - test: 91.18%  (2687/2947)
Ensemble of 4 - test 92.70%  (2732/2947)

Epoch 1000: Training Accuracy 7312/7352=99.46%
Model 5/9 - test: 92.47%  (2725/2947)
Ensemble of 5 - test 93.65%  (2760/2947)

Epoch 1000: Training Accuracy 7289/7352=99.14%
Model 6/9 - test: 91.28%  (2690/2947)
Ensemble of 6 - test 93.01%  (2741/2947)

Epoch 1000: Training Accuracy 7274/7352=98.94%
Model 7/9 - test: 91.62%  (2700/2947)
Ensemble of 7 - test 93.65%  (2760/2947)

Epoch 1000: Training Accuracy 7289/7352=99.14%
Model 8/9 - test: 91.31%  (2691/2947)
Ensemble of 8 - test 93.35%  (2751/2947)

Epoch 1000: Training Accuracy 7301/7352=99.31%
Model 9/9 - test: 92.40%  (2723/2947)
Ensemble of 9 - test 93.65%  (2760/2947)

147.43s user 8.23s system 564% cpu 27.577 total
```

References
----------
* [1] [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
* [2] ["A Public Domain Dataset for Human Activity Recognition using Smartphones", D. Anguita, A. Ghio, L. Oneto, X. Parra, Jorge Luis Reyes-Ortiz, The European Symposium on Artificial Neural Networks, 2013](https://www.semanticscholar.org/paper/A-Public-Domain-Dataset-for-Human-Activity-using-Anguita-Ghio/83de43bc849ad3d9579ccf540e6fe566ef90a58e)
* [3] ["Online Passive-Aggressive Algorithms", Koby Crammer et al, Journal of Machine Learning Research 7 (2006) 551–585, 2006](https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
* [4] [Learning Vector Quantization](https://en.wikipedia.org/wiki/Learning_vector_quantization)

