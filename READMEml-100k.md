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
cargo run --example movielens100k --release -- --data DATA/ml-100k --dim 8192  --topk 10 --split a

Split ua  | 90570 train, 9430 test ratings, 1682 movies, dim=8192

Popularity Recommender
Top-10 Hit Rate: 52.46%  (490/934)
Precision@10: 0.0745
Recall@10: 0.1346

HyperVector Profile Recommender
Top-10 Hit Rate: 72.81%  (680/934)
Precision@10: 0.1355
Recall@10: 0.2454

── Top-10 recommendations for user 1 ──
  # 1  movie   49  Star Wars (1977)
  # 2  movie    8  Dead Man Walking (1995)
  # 3  movie  180  Return of the Jedi (1983)
  # 4  movie  474  Trainspotting (1996)
  # 5  movie    6  Twelve Monkeys (1995)
  # 6  movie  507  People vs. Larry Flynt, The (1996)
  # 7  movie  221  Star Trek: First Contact (1996)
  # 8  movie   14  Mr. Holland's Opus (1995)
  # 9  movie  116  Rock, The (1996)
  #10  movie  514  Boot, Das (1981)
cargo run --example movielens100k --release -- --data DATA/ml-100k --dim 8192  0.53s user 0.09s system 48% cpu 1.272 total
```

## Experiments

We use the official splits for the data set and score only the movies that were liked (threshold >=4).
Two models are evaluated: 1) Simple Popular movies recommender + 2) HyperVector Profile

### Model: Training Popularity (Most popular unseen movies in the training set).

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          79%  |       0.22   |    0.11   |
|    2  |          70%  |       0.18   |    0.12   |
|    3  |          61%  |       0.13   |    0.12   |
|    4  |          62%  |       0.14   |    0.13   |
|    5  |          61%  |       0.13   |    0.13   |
|    a  |          52%  |       0.07   |    0.13   |
|    b  |          49%  |       0.07   |    0.12   |

### Model: HyperVector Profile

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          88%  |       0.32   |    0.20   |
|    2  |          86%  |       0.26   |    0.22   |
|    3  |          81%  |       0.22   |    0.24   |
|    4  |          77%  |       0.21   |    0.23   |
|    5  |          76%  |       0.20   |    0.24   |
|    a  |          73%  |       0.14   |    0.25   |
|    b  |          72%  |       0.13   |    0.24   |


## References

1. [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

