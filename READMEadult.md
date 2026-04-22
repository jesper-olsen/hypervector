# Hypervector - UCI Adult / Census Income

Classification on the Adult dataset [1]:

> Predict whether annual income of an individual exceeds $50K/yr based on census data. 

## Modelling


## Usage

```bash
cargo run --example adult -- --help
Usage: adult [OPTIONS]

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

Download the dataset from [1] and place the two data files in:
```bash
DATA/ADULT/adult_test.csv  
DATA/ADULT/adult_train.csv
```

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

Training an ensemble of 5 models.

Each model has 4 prototype hypervectors per class (two classes).

Each hypervector is a 16384 dimensional binary vector - taking up 2kb.

LVQ trainer [2].

The ensemble achieves 85% classification accuracy. 

See [here](https://github.com/jesper-olsen/decision_tree) for a decision tree solution on the same task.

```bash
cargo run --release --example adult -- --dim 16384 --trainer lvq  --prototypes 4 --ensemble-size 5 --mode binary
Epoch 1000: Training Accuracy 27509/32561=84.48%
Model 1/5 - test: 84.51%  (13759/16281)

Epoch 1000: Training Accuracy 27341/32561=83.97%
Model 2/5 - test: 84.16%  (13702/16281)

Epoch 1000: Training Accuracy 27665/32561=84.96%
Model 3/5 - test: 84.44%  (13748/16281)
Ensemble of 3 - test 84.61%  (13776/16281)

Epoch 1000: Training Accuracy 27706/32561=85.09%
Model 4/5 - test: 84.83%  (13811/16281)
Ensemble of 4 - test 84.76%  (13799/16281)

Epoch 1000: Training Accuracy 27655/32561=84.93%
Model 5/5 - test: 85.15%  (13864/16281)
Ensemble of 5 - test 85.07%  (13851/16281)

Model accuracies - avg 84.62%, min 84.16%, max 85.15
```



## References

1. [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)
2. T. Kohonen, "Improved versions of learning vector quantization", IJCNN 1990.

