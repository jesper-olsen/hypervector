# Hypervector - MovieLens 100k

Movie recommendation on the MovieLens 100k dataset [1].


## Modelling

HyperVector Profile - 3-step training:

1. Movies are assigned a unique random hypervector.
2. Users are assigned a bundling of the movies (hypervectors) they have rated positively.
3. Movies are assigned a (weighted) bundling of the users that have rated the movies positively.


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

Download the dataset from [1] and unzip it so the path `DATA/ml-100k/u.data` exists.

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --example movielens100k --release -- --data DATA/ml-100k --dim 8192  --topk 10 --split a

Split ua  |  90570 train, 9430 test ratings, 1682 movies, dim=8192

Popularity Recommender
Top-10 Hit Rate: 52.46%  (490/934)
Precision@10: 0.0745
Recall@10: 0.1346

HyperVector Profile Recommender
Top-10 Hit Rate: 69.38%  (648/934)
Precision@10: 0.1229
Recall@10: 0.2185

── Top-10 recommendations for user 1 ──
  # 1  movie  654  Stand by Me (1986)
  # 2  movie  650  Glory (1989)
  # 3  movie  422  E.T. the Extra-Terrestrial (1982)
  # 4  movie  495  It's a Wonderful Life (1946)
  # 5  movie  526  Gandhi (1982)
  # 6  movie  366  Clueless (1995)
  # 7  movie  317  Schindler's List (1993)
  # 8  movie  482  Casablanca (1942)
  # 9  movie  201  Groundhog Day (1993)
  #10  movie  356  One Flew Over the Cuckoo's Nest (1975)
```

## Experiments

Model: Training Popularity (Most popular unseen movies in the training set).

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          79%  |       0.22   |    0.11   |
|    2  |          70%  |       0.18   |    0.12   |
|    3  |          61%  |       0.13   |    0.12   |
|    4  |          62%  |       0.14   |    0.13   |
|    5  |          61%  |       0.13   |    0.13   |
|    a  |          52%  |       0.07   |    0.13   |
|    b  |          49%  |       0.07   |    0.12   |

Model: HyperVector Profile

| Split | Top-10 Hit    | Precision@10 | Recall@10 |
| ----: | -------------:|-------------:|---------: |
|    1  |          86%  |       0.31   |    0.18   |
|    2  |          84%  |       0.25   |    0.21   |
|    3  |          77%  |       0.21   |    0.23   |
|    4  |          75%  |       0.20   |    0.22   |
|    5  |          75%  |       0.19   |    0.23   |
|    a  |          69%  |       0.12   |    0.22   |
|    b  |          69%  |       0.12   |    0.22   |


## References

1. [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

