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
Top-10 Hit Rate: 60.60%  (566/934)
Precision@10: 0.0966
Recall@10: 0.1744

── Top-10 recommendations for user 1 ──
  # 1  movie  655  Stand by Me (1986)
  # 2  movie  496  It's a Wonderful Life (1946)
  # 3  movie  367  Clueless (1995)
  # 4  movie  715  To Die For (1995)
  # 5  movie  549  Rob Roy (1995)
  # 6  movie  588  Beauty and the Beast (1991)
  # 7  movie  518  Miller's Crossing (1990)
  # 8  movie  693  Casino (1995)
  # 9  movie  498  African Queen, The (1951)
  #10  movie  431  Highlander (1986)
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
|    1  |          83%  |       0.27   |    0.15   |
|    2  |          80%  |       0.23   |    0.18   |
|    3  |          75%  |       0.19   |    0.20   |
|    4  |          71%  |       0.17   |    0.19   |
|    5  |          72%  |       0.17   |    0.21   |
|    a  |          61%  |       0.10   |    0.17   |
|    b  |          62%  |       0.10   |    0.18   |


## References

1. [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

