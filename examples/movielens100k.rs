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
const N_MOVIES: usize = 1682;

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
    fn get_user(&self, uid: usize) -> (&[u32], &[u8]) {
        let start = self.offsets[uid];
        let end = self.offsets[uid + 1];
        (&self.movies[start..end], &self.scores[start..end])
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

/// Pre-computed per-user sets needed by both evaluators.
struct EvalSets {
    /// Movies each user rated in the training set (seen, to be excluded from recs).
    train: Vec<HashSet<u32>>,
    /// Movies each user liked (score >= threshold) in the test set (ground truth).
    test: Vec<HashSet<u32>>,
}

impl EvalSets {
    fn build(ratings_train: &Ratings, ratings_test: &Ratings, threshold: u8) -> Self {
        let mut train: Vec<HashSet<u32>> = (0..N_USERS).map(|_| HashSet::new()).collect();
        let mut test: Vec<HashSet<u32>> = (0..N_USERS).map(|_| HashSet::new()).collect();

        for user in 0..N_USERS {
            let (movies, _scores) = ratings_train.get_user(user);
            for m in movies {
                train[user].insert(*m);
            }
            let (movies, scores) = ratings_test.get_user(user);
            for i in 0..movies.len() {
                if scores[i] >= threshold {
                    test[user].insert(movies[i]);
                }
            }
        }
        Self { train, test }
    }
}

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
fn build_item_hdvs<H, R>(ratings: &Ratings, threshold: u8, rng: &mut R) -> Vec<H>
where
    H: HyperVector,
    R: Rng + ?Sized,
{
    // ── Step 1: random seed HDV per movie ───────────────────────────────────
    let seed_hdvs: Vec<H> = (0..N_MOVIES).map(|_| H::random(rng)).collect();

    // Pre-calculate movie popularity (for IDF)
    let mut movie_popularity = vec![0usize; N_MOVIES];
    for i in 0..ratings.scores.len() {
        if ratings.scores[i] >= threshold {
            movie_popularity[ratings.movies[i] as usize] += 1;
        }
    }

    // ── Step 2: user HDV = bundle of seed HDVs of movies they liked ─────────
    let mut user_accs: Vec<H::Accumulator> = (0..N_USERS).map(|_| H::Accumulator::new()).collect();
    for user in 0..N_USERS {
        let (movies, _scores) = ratings.get_user(user);
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
        (0..N_MOVIES).map(|_| H::Accumulator::new()).collect();
    for user in 0..N_USERS {
        let (movies, scores) = ratings.get_user(user);
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
    (0..N_MOVIES)
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

    let (movies, scores) = ratings.get_user(user);

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
fn rank_movies<H: HyperVector + Sync>(
    profile: &H,
    item_hdvs: &[H],
    exclude: &HashSet<u32>,
) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = item_hdvs
        .iter()
        .enumerate()
        .filter(|(id, _)| !exclude.contains(&(*id as u32)))
        .map(|(id, hdv)| (id as u32, profile.distance(hdv)))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

// ── Popularity baseline ───────────────────────────────────────────────────────

/// Pre-sort all movies by training popularity once, then per-user we only filter.
fn build_popularity_ranking(train: &Ratings, threshold: u8) -> Vec<u32> {
    let mut counts: Vec<usize> = vec![0usize; N_MOVIES];
    for i in 0..train.movies.len() {
        if train.scores[i] >= threshold {
            counts[train.movies[i] as usize] += 1
        }
    }
    let mut ranking: Vec<u32> = (0..N_MOVIES).map(|i| i as u32).collect();
    ranking.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]));
    ranking
}

fn evaluate_pop(sets: &EvalSets, train: &Ratings, args: &Args) {
    // Compute global popularity ranking once — O(movies log movies).
    // Per-user we only filter out seen items from this pre-sorted list.
    let global_ranking = build_popularity_ranking(train, args.threshold);

    let mut precision_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut users = 0usize;
    let mut users_with_hits = 0usize;

    for user in 0..sets.test.len() {
        let relevant_items = &sets.test[user];
        if relevant_items.is_empty() {
            continue;
        }
        let seen = sets.train.get(user).cloned().unwrap_or_default();

        let topk: Vec<u32> = global_ranking
            .iter()
            .filter(|id| !seen.contains(id))
            .take(args.topk)
            .cloned()
            .collect();

        let hits = topk.iter().filter(|id| relevant_items.contains(id)).count();
        precision_sum += hits as f64 / args.topk as f64;
        recall_sum += hits as f64 / relevant_items.len() as f64;
        users += 1;
        if hits > 0 {
            users_with_hits += 1;
        }
    }

    print_metrics(
        "Popularity Recommender",
        args.topk,
        users_with_hits,
        users,
        precision_sum,
        recall_sum,
    );
}

// ── HDC Evaluation ────────────────────────────────────────────────────────────
//
// Metrics:
//   Hit Rate:    What percentage of users saw at least one movie they liked?
//   Recall@K:    Out of everything the user liked, what fraction did we find?
//   Precision@K: Out of K slots, how many were actually useful?

fn evaluate<H: HyperVector + Sync>(sets: &EvalSets, train: &Ratings, item_hdvs: &[H], args: &Args) {
    let mut precision_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut users = 0usize;
    let mut users_with_hits = 0usize;
    let mut skipped = 0usize;

    for user in 0..sets.test.len() {
        let relevant_items = &sets.test[user];
        if relevant_items.is_empty() {
            continue;
        }
        let profile = match build_profile(user, train, item_hdvs, args.threshold) {
            Some(p) => p,
            None => {
                skipped += relevant_items.len();
                continue;
            }
        };

        let Some(seen) = sets.train.get(user) else {
            // no training data - doesn't happen if profile is calculated...
            skipped += relevant_items.len();
            continue;
        };
        let ranked = rank_movies(&profile, item_hdvs, seen);

        let topk: Vec<u32> = ranked.iter().take(args.topk).cloned().collect();
        let hits = topk.iter().filter(|id| relevant_items.contains(id)).count();

        precision_sum += hits as f64 / args.topk as f64;
        recall_sum += hits as f64 / relevant_items.len() as f64;
        users += 1;
        if hits > 0 {
            users_with_hits += 1;
        }
    }

    print_metrics(
        "HyperVector Profile Recommender",
        args.topk,
        users_with_hits,
        users,
        precision_sum,
        recall_sum,
    );

    if skipped > 0 {
        println!("  ({skipped} skipped: user had no liked training movies)");
    }
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

    let (movies, _scores) = train.get_user(user);

    let seen: HashSet<_> = movies.iter().map(|&m| m).collect();

    let ranked = rank_movies(&profile, item_hdvs, &seen);
    for (rank, &movie_id) in ranked.iter().take(args.topk).enumerate() {
        let title = &titles[movie_id as usize];
        println!("  #{:2}  movie {movie_id:4}  {title}", rank + 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run<H: HyperVector + Sync>(args: &Args) {
    let split = &args.split;
    //let train = load_ratings(&args.data.join(format!("u{split}.base")));
    //let test = load_ratings(&args.data.join(format!("u{split}.test")));
    let titles = load_titles(&args.data);
    assert!(N_MOVIES == titles.len());

    let train = Ratings::load(&args.data.join(format!("u{split}.base")));
    let test = Ratings::load(&args.data.join(format!("u{split}.test")));

    println!(
        "Split u{split}  | {} train, {} test ratings, {} movies, dim={}",
        train.movies.len(),
        test.movies.len(),
        titles.len(),
        args.dim
    );
    println!();

    // Build the per-user train/test sets once; both evaluators share them.
    let sets = EvalSets::build(&train, &test, args.threshold);

    evaluate_pop(&sets, &train, args);

    let mut rng = MersenneTwister64::new(42);
    let item_hdvs: Vec<H> = build_item_hdvs(&train, args.threshold, &mut rng);

    evaluate(&sets, &train, &item_hdvs, args);
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
