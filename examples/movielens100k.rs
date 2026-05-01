/// MovieLens 100K – Hyperdimensional Profile Recommendation
///
/// Encoding strategy (collaborative):
///   - Each USER gets a random seed HDV.
///   - Each MOVIE's HDV = bundle of the HDVs of users who liked it.
///     Two movies liked by overlapping audiences end up geometrically close.
///   - A user's PROFILE = bundle of the HDVs of movies they liked.
///   - Recommendation: rank unseen movies by distance(movie_hdv, user_profile).
///
/// Evaluation: u{split}.base / u{split}.test
///   For every liked (rating >= threshold) item in the test set, check whether
///   it appears in the user's top-K recommendations built from the base set.
use clap::Parser;
use hypervector::hdv;
use hypervector::types::binary::Binary;
use hypervector::types::traits::{Accumulator, HyperVector, UnitAccumulator};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
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

type UserId = u32;
type MovieId = u32;

#[derive(Debug, Clone)]
struct Rating {
    user: UserId,
    movie: MovieId,
    score: u8,
}

fn load_ratings(path: &Path) -> Vec<Rating> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path:?}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut it = line.split_whitespace();
            let user = it.next().unwrap().parse().unwrap();
            let movie = it.next().unwrap().parse::<u32>().unwrap() - 1; // convert 1-based to 0-based
            let score = it.next().unwrap().parse().unwrap();
            Rating { user, movie, score }
        })
        .collect()
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
            let id: MovieId = it.next().unwrap().parse().unwrap();
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
    train: Vec<HashSet<MovieId>>,
    /// Movies each user liked (score >= threshold) in the test set (ground truth).
    test: Vec<HashSet<MovieId>>,
}

impl EvalSets {
    fn build(
        ratings_train: &[Rating],
        ratings_test: &[Rating],
        threshold: u8,
        n_users: usize,
    ) -> Self {
        let mut train: Vec<HashSet<MovieId>> = (0..n_users).map(|_| HashSet::new()).collect();
        let mut test: Vec<HashSet<MovieId>> = (0..n_users).map(|_| HashSet::new()).collect();

        for r in ratings_train {
            train[r.user as usize].insert(r.movie);
        }
        for r in ratings_test {
            if r.score >= threshold {
                test[r.user as usize].insert(r.movie);
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
fn build_item_hdvs<H, R>(
    ratings: &[Rating],
    n_movies: usize,
    n_users: usize,
    threshold: u8,
    rng: &mut R,
) -> Vec<H>
where
    H: HyperVector,
    R: Rng + ?Sized,
{
    // ── Step 1: random seed HDV per movie ───────────────────────────────────
    let seed_hdvs: Vec<H> = (0..n_movies).map(|_| H::random(rng)).collect();

    // ── Step 2: user HDV = bundle of seed HDVs of movies they liked ─────────
    let mut user_accs: Vec<H::UnitAccumulator> =
        (0..n_users).map(|_| H::UnitAccumulator::new()).collect();
    for r in ratings.iter().filter(|r| r.score >= threshold) {
        user_accs[r.user as usize].add(&seed_hdvs[r.movie as usize]);
    }
    let user_hdvs: Vec<H> = user_accs.iter_mut().map(|acc| acc.finalize()).collect();

    // ── Step 3: movie HDV = weighted bundle of user HDVs ────────────────────
    let mut movie_accs: Vec<H::Accumulator> =
        (0..n_movies).map(|_| H::Accumulator::new()).collect();
    for r in ratings.iter().filter(|r| r.score >= threshold) {
        let count = user_accs[r.user as usize].count();
        if count > 0 {
            let weight = 1.0 / (count as f64).sqrt();
            movie_accs[r.movie as usize].add(&user_hdvs[r.user as usize], weight);
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
    user: UserId,
    ratings: &[Rating],
    item_hdvs: &[H],
    threshold: u8,
) -> Option<H> {
    let mut acc = H::UnitAccumulator::default();

    for r in ratings {
        if r.user == user && r.score >= threshold {
            acc.add(&item_hdvs[r.movie as usize]);
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
fn rank_movies<H: HyperVector>(
    profile: &H,
    item_hdvs: &[H],
    exclude: &HashSet<MovieId>,
) -> Vec<MovieId> {
    let mut scored: Vec<(MovieId, f32)> = item_hdvs
        .iter()
        .enumerate()
        .filter(|(id, _)| !exclude.contains(&(*id as u32)))
        .map(|(id, hdv)| (id as MovieId, profile.distance(hdv)))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

// ── Popularity baseline ───────────────────────────────────────────────────────

/// Pre-sort all movies by training popularity once, then per-user we only filter.
fn build_popularity_ranking(train: &[Rating], threshold: u8, n_movies: usize) -> Vec<MovieId> {
    let mut counts: Vec<usize> = vec![0usize; n_movies];
    for r in train {
        if r.score >= threshold {
            counts[r.movie as usize] += 1
        }
    }
    let mut ranking: Vec<MovieId> = (0..n_movies).map(|i| i as MovieId).collect();
    ranking.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]));
    ranking
}

fn evaluate_pop(sets: &EvalSets, train: &[Rating], n_movies: usize, args: &Args) {
    // Compute global popularity ranking once — O(movies log movies).
    // Per-user we only filter out seen items from this pre-sorted list.
    let global_ranking = build_popularity_ranking(train, args.threshold, n_movies);

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

        let topk: Vec<MovieId> = global_ranking
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

fn evaluate<H: HyperVector>(sets: &EvalSets, train: &[Rating], item_hdvs: &[H], args: &Args) {
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
        let profile =
            match build_profile(user.try_into().unwrap(), train, item_hdvs, args.threshold) {
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

        let topk: Vec<MovieId> = ranked.iter().take(args.topk).cloned().collect();
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

fn demo_user<H: HyperVector>(
    user: UserId,
    train: &[Rating],
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

    let seen: HashSet<_> = train
        .iter()
        .filter(|r| r.user == user)
        .map(|r| r.movie)
        .collect();

    let ranked = rank_movies(&profile, item_hdvs, &seen);
    for (rank, &movie_id) in ranked.iter().take(args.topk).enumerate() {
        let title = &titles[movie_id as usize];
        println!("  #{:2}  movie {movie_id:4}  {title}", rank + 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run<H: HyperVector>(args: &Args) {
    let split = &args.split;
    let train = load_ratings(&args.data.join(format!("u{split}.base")));
    let test = load_ratings(&args.data.join(format!("u{split}.test")));
    let titles = load_titles(&args.data);
    let n_movies = titles.len();

    println!(
        "Split u{split}  |  {} train, {} test ratings, {} movies, dim={}",
        train.len(),
        test.len(),
        n_movies,
        args.dim
    );
    println!();

    // Build the per-user train/test sets once; both evaluators share them.
    let n_users = train.iter().map(|r| r.user).max().unwrap() as usize + 1;
    let sets = EvalSets::build(&train, &test, args.threshold, n_users);

    evaluate_pop(&sets, &train, n_movies, args);

    let mut rng = MersenneTwister64::new(42);
    let item_hdvs: Vec<H> = build_item_hdvs(&train, n_movies, n_users, args.threshold, &mut rng);

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
