# Hypervector - MovieLens 100k

Movie recommendation on the MovieLens 100k dataset [1].


## Modelling

HyperVector Profile - 3-step training:

1. Each movie is assigned a random seed hypervector.
2. Each user is represented as an IDF-weighted bundle of the seed HDVs of movies they liked.
3. Each movie's final HDV is a bundle of the HDVs of users who liked it,
   weighted by inverse sqrt of user activity to prevent prolific raters dominating.


## Usage

```bash
cargo run --example movielens100k --release -- --help

MovieLens 100K profile recommendation with HDC

Usage: movielens100k [OPTIONS]

Options:
      --data <DATA>            Path to the ml-100k directory [default: ml-100k]
      --dim <DIM>              Dimension (binary HDV) [default: 4096]
      --threshold <THRESHOLD>  Minimum rating to treat as "liked" (1-5) [default: 4]
      --topk <TOPK>            Top-K for hit-rate evaluation [default: 10]
      --split <SPLIT>          Which split to use (1-5, or 'a' / 'b') [default: 1]
  -h, --help                   Print help
```

## Preliminaries

Download the dataset from [1] and unzip it so the path `DATA/ml-100k/` exists.

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --example movielens100k --release -- --data DATA/ml-100k --dim 8192  --topk 10 --threshold 3 --split a

Split ua  | 90570 train, 9430 test ratings, 1682 movies, dim=8192

Popularity Recommender
Top-10 Hit Rate: 64.79%  (611/943)
Precision@10: 0.0972
Recall@10: 0.1170

HyperVector Profile Recommender
Top-10 Hit Rate: 78.79%  (743/943)
Precision@10: 0.1730
Recall@10: 0.2101

── Top-10 recommendations for user 1 ──
  # 1  movie    8  Dead Man Walking (1995)
  # 2  movie   49  Star Wars (1977)
  # 3  movie  180  Return of the Jedi (1983)
  # 4  movie  507  People vs. Larry Flynt, The (1996)
  # 5  movie    6  Twelve Monkeys (1995)
  # 6  movie  474  Trainspotting (1996)
  # 7  movie  221  Star Trek: First Contact (1996)
  # 8  movie  741  Ransom (1996)
  # 9  movie   14  Mr. Holland's Opus (1995)
  #10  movie  116  Rock, The (1996)
```

## Experiments

We use the official splits for the data set and score only the movies that were liked (threshold >=3).
Two models are evaluated: 1) Simple Popular movies recommender + 2) HyperVector Profile

### Model: Training Popularity (Most popular unseen movies in the training set).

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          85%  |       0.26   |    0.10   |
|    2  |          80%  |       0.21   |    0.11   |
|    3  |          71%  |       0.16   |    0.11   |
|    4  |          67%  |       0.15   |    0.11   |
|    5  |          61%  |       0.13   |    0.13   |
|    a  |          64%  |       0.10   |    0.12   |
|    b  |          59%  |       0.09   |    0.11   |

### Model: HyperVector Profile

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          92%  |       0.39   |    0.17   |
|    2  |          89%  |       0.32   |    0.19   |
|    3  |          85%  |       0.26   |    0.20   |
|    4  |          77%  |       0.21   |    0.23   |
|    5  |          80%  |       0.25   |    0.20   |
|    a  |          79%  |       0.17   |    0.21   |
|    b  |          79%  |       0.16   |    0.20   |



## References

1. [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

