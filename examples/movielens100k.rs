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
use hypervector::types::traits::{HyperVector, UnitAccumulator};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;
use std::collections::{HashMap, HashSet};
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
            let movie = it.next().unwrap().parse().unwrap();
            let score = it.next().unwrap().parse().unwrap();
            Rating { user, movie, score }
        })
        .collect()
}

/// u.item is Latin-1 encoded – read as bytes and lossily convert.
fn load_titles(data: &Path) -> HashMap<MovieId, String> {
    let bytes = fs::read(data.join("u.item")).unwrap_or_default();
    // Latin-1: every byte is a valid Unicode code point with the same value.
    let text: String = bytes.iter().map(|&b| b as char).collect();
    text.lines()
        .filter_map(|line| {
            let mut it = line.splitn(3, '|');
            let id: MovieId = it.next()?.parse().ok()?;
            let title = it.next()?.to_owned();
            Some((id, title))
        })
        .collect()
}

// ── HDC helpers ──────────────────────────────────────────────────────────────

/// Collaborative item encoding:
///   1. Give each user a random seed HDV.
///   2. Each movie's HDV = bundle of the seed HDVs of users who liked it.
///
/// Movies with no likes in `ratings` fall back to a random HDV so the map
/// is complete over all `movie_ids`.
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
    // Step 1: random HDV per user
    let user_ids: Vec<UserId> = {
        let s: HashSet<UserId> = ratings.iter().map(|r| r.user).collect();
        s.into_iter().collect()
    };
    let user_hdvs: HashMap<UserId, H> = user_ids.iter().map(|&id| (id, H::random(rng))).collect();

    // Step 2: accumulate liked-user HDVs per movie
    let mut accumulators: HashMap<MovieId, H::UnitAccumulator> = HashMap::new();
    for r in ratings {
        if r.score >= threshold
            && let Some(u_hdv) = user_hdvs.get(&r.user)
        {
            accumulators.entry(r.movie).or_default().add(u_hdv);
        }
    }

    // Step 3: finalise; movies with zero likes get a random fallback
    movie_ids
        .iter()
        .map(|&id| {
            let hdv = match accumulators.remove(&id) {
                Some(mut acc) => acc.finalize(),
                None => H::random(rng),
            };
            (id, hdv)
        })
        .collect()
}

/// Build a user profile by bundling the HDVs of all movies the user liked.
fn build_profile<H: HyperVector>(
    user: UserId,
    ratings: &[Rating],
    item_hdvs: &HashMap<MovieId, H>,
    threshold: u8,
) -> Option<H> {
    let mut acc = H::UnitAccumulator::default();

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
fn rank_movies<H: HyperVector>(
    profile: &H,
    item_hdvs: &HashMap<MovieId, H>,
    exclude: &HashSet<MovieId>,
) -> Vec<MovieId> {
    let mut scored: Vec<(MovieId, f32)> = item_hdvs
        .iter()
        .filter(|(id, _)| !exclude.contains(id))
        .map(|(&id, hdv)| (id, profile.distance(hdv)))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

// ── Evaluation ───────────────────────────────────────────────────────────────

fn evaluate<H: HyperVector>(
    train: &[Rating],
    test: &[Rating],
    item_hdvs: &HashMap<MovieId, H>,
    args: &Args,
) {
    // ── Build user → seen movies (from train) ───────────────────────────────
    let mut user_train_movies: HashMap<UserId, HashSet<MovieId>> = HashMap::new();
    for r in train {
        user_train_movies.entry(r.user).or_default().insert(r.movie);
    }

    // ── Build user → liked test items ───────────────────────────────────────
    let mut test_by_user: HashMap<UserId, HashSet<MovieId>> = HashMap::new();
    for r in test {
        if r.score >= args.threshold {
            test_by_user.entry(r.user).or_default().insert(r.movie);
        }
    }

    // ── Metrics ─────────────────────────────────────────────────────────────
    let mut hit_count = 0usize; // for hit-rate (recall-style)
    let mut total_test_items = 0usize;

    let mut precision_sum = 0.0; // for Precision@K (macro average)
    let mut user_count = 0usize;

    let mut skipped = 0usize;

    // ── Per-user evaluation ─────────────────────────────────────────────────
    for (&user, relevant_items) in &test_by_user {
        // Build profile
        let profile = match build_profile(user, train, item_hdvs, args.threshold) {
            Some(p) => p,
            None => {
                skipped += relevant_items.len();
                continue;
            }
        };

        // Movies already seen in training
        let seen = user_train_movies.get(&user).cloned().unwrap_or_default();

        // Rank all unseen movies
        let ranked = rank_movies(&profile, item_hdvs, &seen);

        // Take top-K
        let topk: Vec<MovieId> = ranked.iter().take(args.topk).cloned().collect();

        // ── Precision@K ─────────────────────────────────────────────────────
        let hits_in_topk = topk.iter().filter(|id| relevant_items.contains(id)).count();

        let precision = hits_in_topk as f64 / args.topk as f64;
        precision_sum += precision;
        user_count += 1;

        // ── Hit-rate (your original metric) ─────────────────────────────────
        for movie in relevant_items {
            if topk.contains(movie) {
                hit_count += 1;
            }
            total_test_items += 1;
        }
    }

    // ── Final metrics ───────────────────────────────────────────────────────
    let hit_rate = if total_test_items > 0 {
        100.0 * hit_count as f64 / total_test_items as f64
    } else {
        0.0
    };

    let precision_at_k = if user_count > 0 {
        precision_sum / user_count as f64
    } else {
        0.0
    };

    // ── Output ──────────────────────────────────────────────────────────────
    println!(
        "Top-{} Hit Rate: {hit_rate:.2}%  ({hit_count}/{total_test_items})",
        args.topk
    );

    println!("Precision@{}: {precision_at_k:.4}", args.topk);

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
    println!("\n── Top-{} recommendations for user {user} ──", args.topk);
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
        println!("  #{rnk:2}  movie {movie_id:4}  {title}", rnk = rank + 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run<H: HyperVector>(args: &Args) {
    let split = &args.split;
    let base = load_ratings(&args.data.join(format!("u{split}.base")));
    let test = load_ratings(&args.data.join(format!("u{split}.test")));
    let titles = load_titles(&args.data);

    let movie_ids: Vec<MovieId> = {
        let ids: HashSet<MovieId> = base.iter().chain(test.iter()).map(|r| r.movie).collect();
        let mut v: Vec<_> = ids.into_iter().collect();
        v.sort_unstable();
        v
    };

    println!(
        "Split u{split}  |  {} train, {} test ratings, {} movies, dim={}",
        base.len(),
        test.len(),
        movie_ids.len(),
        args.dim
    );

    let mut rng = MersenneTwister64::new(42);
    let item_hdvs: HashMap<MovieId, H> =
        build_item_hdvs(&base, &movie_ids, args.threshold, &mut rng);

    evaluate(&base, &test, &item_hdvs, args);
    demo_user(1, &base, &item_hdvs, &titles, args);
}

fn main() {
    hdv!(binary, Bin1024, 1024);
    hdv!(binary, Bin2048, 2048);
    hdv!(binary, Bin4096, 4096);
    hdv!(binary, Bin8192, 8192);

    let args = Args::parse();

    match args.dim {
        1024 => run::<Bin1024>(&args),
        2048 => run::<Bin2048>(&args),
        4096 => run::<Bin4096>(&args),
        8192 => run::<Bin8192>(&args),
        d => {
            eprintln!("Unsupported dim {d}. Use one of: 1024 2048 4096 8192");
            std::process::exit(1);
        }
    }
}
