//! MovieLens 100K – Hyperdimensional Profile Recommendation
//!
//! Encoding strategy (collaborative):
//!   - Each USER gets a random seed HDV.
//!   - Each MOVIE's HDV = bundle of the HDVs of users who liked it.
//!     Two movies liked by overlapping audiences end up geometrically close.
//!   - A user's PROFILE = bundle of the HDVs of movies they liked.
//!   - Recommendation: rank unseen movies by distance(movie_hdv, user_profile).
//!
//! Evaluation: u{split}.base / u{split}.test
//!   For every liked (rating >= threshold) item in the test set, check whether
//!   it appears in the user's top-K recommendations built from the base set.

use clap::Parser;
use hypervector::hdv;
use hypervector::types::binary::Binary;
use hypervector::types::traits::{Accumulator, HyperVector, UnitAccumulator};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "MovieLens 100K profile recommendation with HDC")]
struct Args {
    /// Path to the ml-100k directory
    #[arg(long, default_value = "ml-100k")]
    data: PathBuf,

    /// Dimension (binary HDV)
    #[arg(long, default_value_t = 4096)]
    dim: usize,

    /// Minimum rating to treat as "liked" (1-5)
    #[arg(long, default_value_t = 4)]
    threshold: u8,

    /// Top-K for hit-rate evaluation
    #[arg(long, default_value_t = 10)]
    topk: usize,

    /// Which split to use (1-5, or 'a' / 'b')
    #[arg(long, default_value = "1")]
    split: String,
}

// ── Data loading ─────────────────────────────────────────────────────────────

const N_USERS: usize = 943;

// Ratings for user u
// let start = offsets[u];
// let end = offsets[u + 1];
//
// for i in start..end {
//     let movie = movies[i];
//     let score = scores[i];
// }

struct Ratings {
    offsets: Vec<usize>,
    movies: Vec<u32>,
    scores: Vec<u8>,
}

impl Ratings {
    fn n_users(&self) -> usize {
        self.offsets.len() - 1
    }

    fn get_ratings(&self, uid: usize) -> (&[u32], &[u8]) {
        let start = self.offsets[uid];
        let end = self.offsets[uid + 1];
        (&self.movies[start..end], &self.scores[start..end])
    }

    fn rated(&self, uid: usize, threshold: u8) -> HashSet<u32> {
        let (movies, scores) = self.get_ratings(uid);
        movies
            .iter()
            .zip(scores.iter())
            .filter(|(_, s)| **s >= threshold)
            .map(|(m, _)| *m)
            .collect()
    }

    fn load(path: &Path) -> Ratings {
        let text = fs::read_to_string(path).unwrap();
        let n = text.lines().filter(|l| !l.is_empty()).count();

        let mut offsets = vec![0usize; N_USERS + 1];
        let mut movies = vec![0u32; n];
        let mut scores = vec![0u8; n];
        let mut current_user = 0;

        for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
            let mut it = line.split_whitespace();
            let user = it.next().unwrap().parse::<usize>().unwrap() - 1;
            movies[i] = it.next().unwrap().parse::<u32>().unwrap() - 1;
            scores[i] = it.next().unwrap().parse::<u8>().unwrap();

            while current_user < user {
                offsets[current_user + 1] = i;
                current_user += 1;
            }
        }
        while current_user < N_USERS {
            offsets[current_user + 1] = n;
            current_user += 1;
        }

        Self {
            offsets,
            movies,
            scores,
        }
    }
}

/// u.item is Latin-1 encoded – read as bytes and lossily convert.
fn load_titles(data: &Path) -> Vec<String> {
    let bytes = fs::read(data.join("u.item")).unwrap_or_default();
    // Latin-1: every byte is a valid Unicode code point with the same value.
    let text: String = bytes.iter().map(|&b| b as char).collect();
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            let mut it = line.splitn(3, '|');
            let id: u32 = it.next().unwrap().parse().unwrap();
            let title = it.next().unwrap().to_owned();
            assert_eq!(id, (i + 1) as u32, "{data:?}/u.item: non-contiguous ID");
            title
        })
        .collect()
}

// ── Evaluation helpers ────────────────────────────────────────────────────────

fn print_metrics(
    label: &str,
    topk: usize,
    hits: usize,
    users: usize,
    precision_sum: f64,
    recall_sum: f64,
) {
    let hit_rate = 100.0 * hits as f64 / users as f64;

    let (precision, recall) = if users > 0 {
        (precision_sum / users as f64, recall_sum / users as f64)
    } else {
        (0.0, 0.0)
    };

    println!("{label}");
    println!("Top-{topk} Hit Rate: {hit_rate:.2}%  ({hits}/{users})");
    println!("Precision@{topk}: {precision:.4}");
    println!("Recall@{topk}: {recall:.4}");
    println!();
}

// ── HDC helpers ──────────────────────────────────────────────────────────────

/// Collaborative encoding:
///
///   Step 1 — Random seed HDV per movie.
///   Step 2 — User HDV = bundle of seed HDVs of movies the user liked.
///             (captures each user's taste as a point in movie-space)
///   Step 3 — Movie HDV = weighted bundle of user HDVs of users who liked it,
///             weight = 1 / (movies liked by that user).
///             Normalising by activity prevents prolific raters from dominating.
fn build_item_hdvs<H, R>(ratings: &Ratings, threshold: u8, n_movies: usize, rng: &mut R) -> Vec<H>
where
    H: HyperVector,
    R: Rng + ?Sized,
{
    // ── Step 1: random seed HDV per movie ───────────────────────────────────
    let seed_hdvs: Vec<H> = (0..n_movies).map(|_| H::random(rng)).collect();

    // Pre-calculate movie popularity (for IDF)
    let mut movie_popularity = vec![0usize; n_movies];
    for i in 0..ratings.scores.len() {
        if ratings.scores[i] >= threshold {
            movie_popularity[ratings.movies[i] as usize] += 1;
        }
    }

    // ── Step 2: user HDV = bundle of seed HDVs of movies they liked ─────────
    let mut user_accs: Vec<H::Accumulator> = (0..N_USERS).map(|_| H::Accumulator::new()).collect();
    for user in 0..N_USERS {
        let (movies, _scores) = ratings.get_ratings(user);
        for i in 0..movies.len() {
            let pop = movie_popularity[movies[i] as usize];
            // IDF weight: rare movies get higher weight
            let idf = 1.0 / (pop as f64).sqrt();
            user_accs[user].add(&seed_hdvs[movies[i] as usize], idf);
        }
    }
    let user_hdvs: Vec<H> = user_accs.iter_mut().map(|acc| acc.finalize()).collect();

    // ── Step 3: movie HDV = weighted bundle of user HDVs ────────────────────
    let mut movie_accs: Vec<H::Accumulator> =
        (0..n_movies).map(|_| H::Accumulator::new()).collect();
    for user in 0..N_USERS {
        let (movies, scores) = ratings.get_ratings(user);
        for i in 0..movies.len() {
            if scores[i] >= threshold {
                let user_signal_strength = user_accs[user].count();
                if user_signal_strength > 0.0 {
                    let weight = 1.0 / (user_signal_strength).sqrt();
                    movie_accs[movies[i] as usize].add(&user_hdvs[user], weight);
                }
            }
        }
    }

    // Movies with zero likes (cold items) fall back to their seed HDV
    (0..n_movies)
        .map(|id| {
            if movie_accs[id].count() > 0.0 {
                movie_accs[id].finalize()
            } else {
                seed_hdvs[id].clone()
            }
        })
        .collect()
}

/// Build a user profile by bundling the HDVs of all movies the user liked.
fn build_profile<H: HyperVector>(
    user: usize,
    ratings: &Ratings,
    item_hdvs: &[H],
    threshold: u8,
) -> Option<H> {
    let mut acc = H::UnitAccumulator::default();

    let (movies, scores) = ratings.get_ratings(user);

    for i in 0..movies.len() {
        if scores[i] >= threshold {
            acc.add(&item_hdvs[movies[i] as usize]);
        }
    }

    if acc.count() == 0 {
        None
    } else {
        Some(acc.finalize())
    }
}

/// Rank movies by similarity to a profile (lower distance = better match).
/// Items in `exclude` (already seen by the user) are omitted.
//fn rank_movies<H: HyperVector + Sync>(
//    profile: &H,
//    item_hdvs: &[H],
//    exclude: &HashSet<u32>,
//) -> Vec<u32> {
//    let mut scored: Vec<(u32, f32)> = item_hdvs
//        .iter()
//        .enumerate()
//        .filter(|(id, _)| !exclude.contains(&(*id as u32)))
//        .map(|(id, hdv)| (id as u32, profile.distance(hdv)))
//        .collect();
//
//    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
//    scored.into_iter().map(|(id, _)| id).collect()
//}

#[derive(PartialEq)]
struct ScoredMovie {
    id: u32,
    distance: f32,
}

// We implement Ord such that the "greatest" element has the largest distance.
// This makes it a Max-Heap based on distance.
impl Eq for ScoredMovie {}

impl PartialOrd for ScoredMovie {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for ScoredMovie {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Rank movies by similarity to a profile (lower distance = better match).
/// Items in `exclude` (already seen by the user) are omitted.
fn rank_movies_topk<H: HyperVector + Sync>(
    profile: &H,
    item_hdvs: &[H],
    exclude: &HashSet<u32>,
    topk: usize,
) -> Vec<u32> {
    let mut heap = BinaryHeap::with_capacity(topk + 1);

    for (id, hdv) in item_hdvs.iter().enumerate() {
        let movie_id = id as u32;
        if exclude.contains(&movie_id) {
            continue;
        }

        heap.push(ScoredMovie {
            id: movie_id,
            distance: profile.distance(hdv),
        });

        // Keep only the topk smallest distances
        if heap.len() > topk {
            heap.pop();
        }
    }

    // The heap now contains the topk closest movies, but in max-distance order.
    // Convert to Vec and reverse so the closest is at index 0.
    heap.into_sorted_vec().into_iter().map(|m| m.id).collect()
}

// ── Popularity baseline ───────────────────────────────────────────────────────

/// Pre-sort all movies by training popularity once, then per-user we only filter.
fn build_popularity_ranking(train: &Ratings, threshold: u8, n_movies: usize) -> Vec<u32> {
    let mut counts: Vec<usize> = vec![0usize; n_movies];
    for i in 0..train.movies.len() {
        if train.scores[i] >= threshold {
            counts[train.movies[i] as usize] += 1
        }
    }
    let mut ranking: Vec<u32> = (0..n_movies).map(|i| i as u32).collect();
    ranking.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]));
    ranking
}

// ── Shared evaluation core ────────────────────────────────────────────────────

/// Evaluate any recommender via a closure.
///
/// `recommend(user)` returns the ranked movie list for that user,
/// or `None` to skip them (e.g. no training data).
fn evaluate_recommender(
    train: &Ratings,
    test: &Ratings,
    threshold: u8,
    label: &str,
    topk: usize,
    mut recommend: impl FnMut(usize) -> Option<Vec<u32>>,
) {
    let mut precision_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut users = 0usize;
    let mut users_with_hits = 0usize;
    let mut skipped = 0usize;

    for user in 0..train.n_users() {
        let relevant_items = test.rated(user, threshold);
        if relevant_items.is_empty() {
            continue;
        }
        let Some(ranked) = recommend(user) else {
            skipped += relevant_items.len();
            continue;
        };

        let hits = ranked
            .iter()
            .take(topk)
            .filter(|id| relevant_items.contains(id))
            .count();
        precision_sum += hits as f64 / topk as f64;
        recall_sum += hits as f64 / relevant_items.len() as f64;
        users += 1;
        if hits > 0 {
            users_with_hits += 1;
        }
    }

    print_metrics(
        label,
        topk,
        users_with_hits,
        users,
        precision_sum,
        recall_sum,
    );

    if skipped > 0 {
        println!("  ({skipped} skipped)");
    }
}

fn evaluate_pop(train: &Ratings, test: &Ratings, args: &Args, n_movies: usize) {
    let global_ranking = build_popularity_ranking(train, args.threshold, n_movies);

    evaluate_recommender(
        train,
        test,
        args.threshold,
        "Popularity Recommender",
        args.topk,
        |user| {
            let seen = train.rated(user, args.threshold);
            Some(
                global_ranking
                    .iter()
                    .filter(|id| !seen.contains(id))
                    .cloned()
                    .collect(),
            )
        },
    );
}

fn evaluate<H: HyperVector + Sync>(train: &Ratings, test: &Ratings, item_hdvs: &[H], args: &Args) {
    evaluate_recommender(
        train,
        test,
        args.threshold,
        "HyperVector Profile Recommender",
        args.topk,
        |user| {
            let profile = build_profile(user, train, item_hdvs, args.threshold)?;
            let seen = train.rated(user, args.threshold);
            Some(rank_movies_topk(&profile, item_hdvs, &seen, args.topk))
        },
    );
}

// ── Demo ─────────────────────────────────────────────────────────────────────

fn demo_user<H: HyperVector + Sync>(
    user: usize,
    train: &Ratings,
    item_hdvs: &[H],
    titles: &[String],
    args: &Args,
) {
    println!("── Top-{} recommendations for user {user} ──", args.topk);
    let profile = match build_profile(user, train, item_hdvs, args.threshold) {
        Some(p) => p,
        None => {
            println!("  (no liked movies in training set)");
            return;
        }
    };

    let (movies, _scores) = train.get_ratings(user);

    let seen: HashSet<_> = movies.iter().map(|&m| m).collect();

    let ranked = rank_movies_topk(&profile, item_hdvs, &seen, args.topk);
    for (rank, &movie_id) in ranked.iter().take(args.topk).enumerate() {
        let title = &titles[movie_id as usize];
        println!("  #{:2}  movie {movie_id:4}  {title}", rank + 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run<H: HyperVector + Sync>(args: &Args) {
    let split = &args.split;
    let titles = load_titles(&args.data);

    let train = Ratings::load(&args.data.join(format!("u{split}.base")));
    let test = Ratings::load(&args.data.join(format!("u{split}.test")));

    println!(
        "Split u{split}  | {} train, {} test ratings, {} movies, dim={}\n",
        train.movies.len(),
        test.movies.len(),
        titles.len(),
        args.dim
    );

    evaluate_pop(&train, &test, args, titles.len());

    let mut rng = MersenneTwister64::new(42);
    let item_hdvs: Vec<H> = build_item_hdvs(&train, args.threshold, titles.len(), &mut rng);

    evaluate(&train, &test, &item_hdvs, args);
    demo_user(1, &train, &item_hdvs, &titles, args);
}

fn main() {
    hdv!(binary, Bin1024, 1024);
    hdv!(binary, Bin2048, 2048);
    hdv!(binary, Bin4096, 4096);
    hdv!(binary, Bin8192, 8192);
    hdv!(binary, Bin16384, 16384);

    let args = Args::parse();

    match args.dim {
        1024 => run::<Bin1024>(&args),
        2048 => run::<Bin2048>(&args),
        4096 => run::<Bin4096>(&args),
        8192 => run::<Bin8192>(&args),
        16384 => run::<Bin16384>(&args),
        d => {
            eprintln!("Unsupported dim {d}. Use one of: 1024 2048 4096 8192");
            std::process::exit(1);
        }
    }
}
