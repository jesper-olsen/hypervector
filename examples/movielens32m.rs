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
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "MovieLens 100K profile recommendation with HDC")]
struct Args {
    /// Path to the ml-32m directory
    #[arg(long, default_value = "ml-100k")]
    data: PathBuf,

    /// Dimension (binary HDV)
    #[arg(long, default_value_t = 4096)]
    dim: usize,

    /// Minimum rating to treat as "liked" (1-10)
    #[arg(long, default_value_t = 8)]
    threshold: u8,

    /// Top-K for hit-rate evaluation
    #[arg(long, default_value_t = 10)]
    topk: usize,

    /// Train/test split, e.g. 0.8=> 80% training, 20% test
    #[arg(long, default_value_t = 0.8)]
    split: f32,
}

// ── Data loading ─────────────────────────────────────────────────────────────

type UserId = u32;
type MovieId = u32;

#[derive(Debug, Clone)]
struct Rating {
    user: UserId,
    movie: MovieId,
    score: u8, // ml-32m uses floats (0.5 increments) - here mapped to 1-10
}

#[derive(Debug, Clone)]
struct RatingWithTs {
    user: UserId,
    movie: MovieId,
    score: u8,
    timestamp: u64,
}

/// Load all ratings from ml-32m/ratings.csv, keeping timestamps for splitting.
fn load_ratings_csv(path: &Path) -> Vec<RatingWithTs> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path:?}: {e}"))
        .lines()
        .skip(1) // skip "userId,movieId,rating,timestamp" header
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut it = line.split(',');
            let user: UserId = it.next().unwrap().parse().unwrap();
            let movie: MovieId = it.next().unwrap().parse().unwrap();
            let score: f32 = it.next().unwrap().parse().unwrap();
            let timestamp: u64 = it.next().unwrap().trim().parse().unwrap();
            let score: u8 = 2 * score as u8;
            RatingWithTs {
                user,
                movie,
                score,
                timestamp,
            }
        })
        .collect()
}

/// Split by timestamp: the oldest `train_fraction` of interactions become
/// train, the rest become test.  Using a global percentile keeps the split
/// deterministic and independent of per-user density.
fn temporal_split(
    mut ratings: Vec<RatingWithTs>,
    train_fraction: f64, // e.g. 0.8
) -> (Vec<Rating>, Vec<Rating>) {
    ratings.sort_unstable_by_key(|r| r.timestamp);

    let cutoff = (ratings.len() as f64 * train_fraction).round() as usize;
    let cutoff_ts = ratings[cutoff.saturating_sub(1)].timestamp;

    let mut train = Vec::new();
    let mut test = Vec::new();

    for r in ratings {
        let rating = Rating {
            user: r.user,
            movie: r.movie,
            score: r.score,
        };
        if r.timestamp <= cutoff_ts {
            train.push(rating);
        } else {
            test.push(rating);
        }
    }
    (train, test)
}

/// movies.csv is plain UTF-8; genres are pipe-separated after the title.
fn load_titles(data: &Path) -> HashMap<MovieId, String> {
    let text = fs::read_to_string(data.join("movies.csv")).unwrap_or_default();

    text.lines()
        .skip(1) // skip "movieId,title,genres" header
        .filter_map(|line| {
            // Split on the *first* two commas only – titles can contain commas.
            let mut it = line.splitn(3, ',');
            let id: MovieId = it.next()?.parse().ok()?;
            let title = it.next()?.to_owned();
            Some((id, title))
        })
        .collect()
}

// ── Evaluation helpers ────────────────────────────────────────────────────────

/// Pre-computed per-user sets needed by both evaluators.
struct EvalSets {
    /// Movies each user rated in the training set (seen, to be excluded from recs).
    user_train_movies: HashMap<UserId, HashSet<MovieId>>,
    /// Movies each user liked (score >= threshold) in the test set (ground truth).
    test_by_user: HashMap<UserId, HashSet<MovieId>>,
}

impl EvalSets {
    fn build(train: &[Rating], test: &[Rating], threshold: u8) -> Self {
        let mut user_train_movies: HashMap<UserId, HashSet<MovieId>> = HashMap::new();
        for r in train {
            user_train_movies.entry(r.user).or_default().insert(r.movie);
        }
        let mut test_by_user: HashMap<UserId, HashSet<MovieId>> = HashMap::new();
        for r in test {
            if r.score >= threshold {
                test_by_user.entry(r.user).or_default().insert(r.movie);
            }
        }
        Self {
            user_train_movies,
            test_by_user,
        }
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
    let (precision, recall) = if users > 0 {
        (precision_sum / users as f64, recall_sum / users as f64)
    } else {
        (0.0, 0.0)
    };

    let hit_rate = 100.0 * hits as f64 / users as f64;
    println!("{label}");
    println!("Top-{topk} Hit Rate: {hit_rate:.2}%  ({hits}/{users})");
    println!("Precision@{topk}: {precision:.4}");
    println!("Recall@{topk}: {recall:.4}");
    println!();
}

// ── HDC helpers ──────────────────────────────────────────────────────────────

/// Iterated collaborative encoding:
///
///   Step 1 — Random seed HDV per movie.
///   Step 2 — User HDV = bundle of seed HDVs of movies the user liked.
///             (captures each user's taste as a point in movie-space)
///   Step 3 — Movie HDV = weighted bundle of user HDVs of users who liked it,
///             weight = 1 / (movies liked by that user).
///             Normalising by activity prevents prolific raters from dominating.
fn build_item_hdvs<H, R>(
    ratings: &[Rating],
    movie_ids: &[MovieId],
    threshold: u8,
    rng: &mut R,
) -> HashMap<MovieId, H>
where
    H: HyperVector,
    R: Rng + ?Sized,
{
    // ── Step 1: random seed HDV per movie ───────────────────────────────────
    let seed_hdvs: HashMap<MovieId, H> = movie_ids.iter().map(|&id| (id, H::random(rng))).collect();

    // ── Step 2: user HDV = bundle of seed HDVs of movies they liked ─────────
    let mut user_accs: HashMap<UserId, H::UnitAccumulator> = HashMap::new();
    for r in ratings {
        if r.score >= threshold
            && let Some(m_hdv) = seed_hdvs.get(&r.movie)
        {
            user_accs.entry(r.user).or_default().add(m_hdv);
        }
    }
    let user_hdvs: HashMap<UserId, H> = user_accs
        .iter_mut()
        .map(|(&id, acc)| (id, acc.finalize()))
        .collect();

    // ── Step 3: movie HDV = weighted bundle of user HDVs ────────────────────
    let mut movie_accs: HashMap<MovieId, H::Accumulator> = HashMap::new();
    for r in ratings {
        if r.score >= threshold
            && let Some(u_hdv) = user_hdvs.get(&r.user)
        {
            let weight = 1.0 / user_accs[&r.user].count().max(1) as f64;
            movie_accs.entry(r.movie).or_default().add(u_hdv, weight);
        }
    }

    // Movies with zero likes (cold items) fall back to their seed HDV
    movie_ids
        .iter()
        .map(|&id| {
            let hdv = match movie_accs.remove(&id) {
                Some(mut acc) => acc.finalize(),
                None => seed_hdvs[&id].clone(),
            };
            (id, hdv)
        })
        .collect()
}

/// Build a user profile by bundling the HDVs of all movies the user liked.
fn build_profile<H>(
    user: UserId,
    ratings: &[Rating],
    item_hdvs: &HashMap<MovieId, H>,
    threshold: u8,
) -> Option<H>
where
    H: HyperVector,
{
    let mut acc = <H as HyperVector>::UnitAccumulator::default();

    for r in ratings {
        if r.user == user
            && r.score >= threshold
            && let Some(hdv) = item_hdvs.get(&r.movie)
        {
            acc.add(hdv);
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
fn rank_movies<H>(
    profile: &H,
    item_hdvs: &HashMap<MovieId, H>,
    exclude: &HashSet<MovieId>,
) -> Vec<MovieId>
where
    H: HyperVector,
{
    let mut scored: Vec<(MovieId, f32)> = item_hdvs
        .iter()
        .filter(|(id, _)| !exclude.contains(id))
        .map(|(&id, hdv)| (id, profile.distance(hdv)))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

// ── Popularity baseline ───────────────────────────────────────────────────────

/// Pre-sort all movies by training popularity once, then per-user we only filter.
fn build_popularity_ranking(
    train: &[Rating],
    threshold: u8,
    all_movie_ids: &[MovieId],
) -> Vec<MovieId> {
    let mut counts: HashMap<MovieId, usize> = HashMap::new();
    for r in train {
        if r.score >= threshold {
            *counts.entry(r.movie).or_default() += 1;
        }
    }
    let mut ranking = all_movie_ids.to_vec();
    ranking.sort_by(|a, b| counts.get(b).unwrap_or(&0).cmp(counts.get(a).unwrap_or(&0)));
    ranking
}

fn evaluate_pop(sets: &EvalSets, train: &[Rating], all_movie_ids: &[MovieId], args: &Args) {
    // Compute global popularity ranking once — O(movies log movies).
    // Per-user we only filter out seen items from this pre-sorted list.
    let global_ranking = build_popularity_ranking(train, args.threshold, all_movie_ids);

    let mut precision_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut user_count = 0usize;
    let mut users_with_hits = 0usize;

    for (user, relevant_items) in &sets.test_by_user {
        let seen = sets
            .user_train_movies
            .get(user)
            .cloned()
            .unwrap_or_default();

        let topk: Vec<MovieId> = global_ranking
            .iter()
            .filter(|id| !seen.contains(id))
            .take(args.topk)
            .cloned()
            .collect();

        let hits = topk.iter().filter(|id| relevant_items.contains(id)).count();
        precision_sum += hits as f64 / args.topk as f64;
        recall_sum += hits as f64 / relevant_items.len() as f64;
        user_count += 1;
        if hits > 0 {
            users_with_hits += 1;
        }
    }

    print_metrics(
        "Popularity Recommender",
        args.topk,
        users_with_hits,
        user_count,
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

fn evaluate<H: HyperVector>(
    sets: &EvalSets,
    train: &[Rating],
    item_hdvs: &HashMap<MovieId, H>,
    args: &Args,
) {
    let mut precision_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut user_count = 0usize;
    let mut users_with_hits = 0usize;
    let mut skipped = 0usize;

    for (user, relevant_items) in &sets.test_by_user {
        let profile = match build_profile(*user, train, item_hdvs, args.threshold) {
            Some(p) => p,
            None => {
                skipped += relevant_items.len();
                continue;
            }
        };

        let seen = sets
            .user_train_movies
            .get(user)
            .cloned()
            .unwrap_or_default();
        let ranked = rank_movies(&profile, item_hdvs, &seen);

        let topk: Vec<MovieId> = ranked.iter().take(args.topk).cloned().collect();
        let hits = topk.iter().filter(|id| relevant_items.contains(id)).count();

        precision_sum += hits as f64 / args.topk as f64;
        recall_sum += hits as f64 / relevant_items.len() as f64;
        user_count += 1;
        if hits > 0 {
            users_with_hits += 1;
        }
    }

    print_metrics(
        "HyperVector Profile Recommender",
        args.topk,
        users_with_hits,
        user_count,
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
    item_hdvs: &HashMap<MovieId, H>,
    titles: &HashMap<MovieId, String>,
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
        let title = titles.get(&movie_id).map(|s| s.as_str()).unwrap_or("?");
        println!("  #{:2}  movie {movie_id:4}  {title}", rank + 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run<H: HyperVector>(args: &Args) {
    let all = load_ratings_csv(&args.data.join("ratings.csv"));
    let (train, test) = temporal_split(all, 0.8);
    let titles = load_titles(&args.data);

    let movie_ids: Vec<MovieId> = {
        let ids: HashSet<MovieId> = train.iter().chain(test.iter()).map(|r| r.movie).collect();
        let mut v: Vec<_> = ids.into_iter().collect();
        v.sort_unstable();
        v
    };

    println!(
        "Split {}  |  {} train, {} test ratings, {} movies, dim={}",
        args.split,
        train.len(),
        test.len(),
        movie_ids.len(),
        args.dim
    );
    println!();

    // Build the per-user train/test sets once; both evaluators share them.
    let sets = EvalSets::build(&train, &test, args.threshold);

    evaluate_pop(&sets, &train, &movie_ids, args);

    let mut rng = MersenneTwister64::new(42);
    let item_hdvs: HashMap<MovieId, H> =
        build_item_hdvs(&train, &movie_ids, args.threshold, &mut rng);

    evaluate(&sets, &train, &item_hdvs, args);
    demo_user(1, &train, &item_hdvs, &titles, args);
}

fn main() {
    hdv!(binary, Bin1024, 1024);
    hdv!(binary, Bin2048, 2048);
    hdv!(binary, Bin4096, 4096);
    hdv!(binary, Bin8192, 8192);
    hdv!(binary, Bin16384, 16384);
    hdv!(binary, Bin32768, 32768);

    let args = Args::parse();

    match args.dim {
        1024 => run::<Bin1024>(&args),
        2048 => run::<Bin2048>(&args),
        4096 => run::<Bin4096>(&args),
        8192 => run::<Bin8192>(&args),
        16384 => run::<Bin16384>(&args),
        32768 => run::<Bin32768>(&args),
        d => {
            eprintln!("Unsupported dim {d}. Use one of: 1024 2048 4096 8192");
            std::process::exit(1);
        }
    }
}
