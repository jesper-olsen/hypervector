use clap::Parser;
use hypervector::hdv;
use hypervector::types::traits::{HyperVector, UnitAccumulator};
use hypervector::types::{binary::Binary, modular::Modular};
use mersenne_twister_rs::MersenneTwister64;
use rand::Rng;

struct Alphabet<H: HyperVector> {
    a: H,
    c: H,
    g: H,
    t: H,
}

impl<H: HyperVector> Alphabet<H> {
    fn new<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            a: H::random(rng),
            c: H::random(rng),
            g: H::random(rng),
            t: H::random(rng),
        }
    }

    fn get(&self, b: u8) -> &H {
        match b {
            b'a' => &self.a,
            b'c' => &self.c,
            b'g' => &self.g,
            b't' => &self.t,
            _ => unreachable!(),
        }
    }
}

// https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences
const DATA_PLUS: [&[u8]; 53] = [
    b"tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt",
    b"tgctatcctgacagttgtcacgctgattggtgtcgttacaatctaacgcatcgccaa",
    b"gtactagagaactagtgcattagcttatttttttgttatcatgctaaccacccggcg",
    b"aattgtgatgtgtatcgaagtgtgttgcggagtagatgttagaatactaacaaactc",
    b"tcgataattaactattgacgaaaagctgaaaaccactagaatgcgcctccgtggtag",
    b"aggggcaaggaggatggaaagaggttgccgtataaagaaactagagtccgtttaggt",
    b"cagggggtggaggatttaagccatctcctgatgacgcatagtcagcccatcatgaat",
    b"tttctacaaaacacttgatactgtatgagcatacagtataattgcttcaacagaaca",
    b"cgacttaatatactgcgacaggacgtccgttctgtgtaaatcgcaatgaaatggttt",
    b"ttttaaatttcctcttgtcaggccggaataactccctataatgcgccaccactgaca",
    b"gcaaaaataaatgcttgactctgtagcgggaaggcgtattatgcacaccccgcgccg",
    b"cctgaaattcagggttgactctgaaagaggaaagcgtaatatacgccacctcgcgac",
    b"gatcaaaaaaatacttgtgcaaaaaattgggatccctataatgcgcctccgttgaga",
    b"ctgcaatttttctattgcggcctgcggagaactccctataatgcgcctccatcgaca",
    b"tttatatttttcgcttgtcaggccggaataactccctataatgcgccaccactgaca",
    b"aagcaaagaaatgcttgactctgtagcgggaaggcgtattatgcacaccgccgcgcc",
# Hypervector - Molecular Biology (Promoter Gene Sequences)

Classification on the UCI Molecular Biology dataset [1] - promoter vs non-promoter.

Using 40,192-dimensional binary hypervectors with multi-scale encoding (bundling 4-, 5-, and 6-grams), we achieve 103/106 correct classifications (97%) under leave-one-out evaluation.

For comparison, the original paper lists these results:

| System	  | Errors | Comments
| :-----      | -----: | :-------
|  KBANN	  |  4/106 | a hybrid ML system
|  BP		  |  8/106 | std backprop with one hidden layer
|  O'Neill    | 12/106 | ad hoc technique from the bio. lit.
|  Near-Neigh | 13/106 | a nearest-neighbor algo (k=3)
|  ID3        | 19/106 | Quinlan's decision-tree builder

Results are not strictly comparable due to differences in evaluation protocols.


## Modelling

Each DNA base (A, C, G, T) is assigned a random hypervector.  
Sequences are encoded by constructing n-gram hypervectors using permutation and binding, 
and bundling them across the sequence.

To improve robustness, encodings for n-1, n, and n+1 are combined (multi-scale representation).

Class prototypes are formed by bundling encoded sequences per class.  
Classification is performed by nearest-prototype similarity.

Evaluation uses leave-one-out cross-validation.


## Usage

```bash
% cargo run --example promoter --release -- --help

Usage: promoter [OPTIONS]

Options:
      --mode <MODE>    [default: binary] [possible values: binary, modular]
      --dim <DIM>      one of 1024, 2048, 4096, 8192, 10048, 20096, 40192 [default: 1024]
      --ngram <NGRAM>  [default: 3]
  -h, --help           Print help
  -V, --version        Print version
```

## Preliminaries

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --release --example promoter -- --dim 40192 --ngram 9  

Mode: binary N-gram: 9 Dim: 40192
2-gram: Accuracy: 78.30% (83/106)
3-gram: Accuracy: 85.85% (91/106)
4-gram: Accuracy: 92.45% (98/106)
5-gram: Accuracy: 97.17% (103/106)
6-gram: Accuracy: 95.28% (101/106)
7-gram: Accuracy: 86.79% (92/106)
8-gram: Accuracy: 83.96% (89/106)
9-gram: Accuracy: 74.53% (79/106)
```

Performance peaks around 5-grams; combining neighboring n-gram scales improves robustness and reduces variance compared to single-scale encodings.

Results are consistent across multiple random seeds (not shown).

## References

1. [UCI Molecular Biology](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences

    b"atgcatttttccgcttgtcttcctgagccgactccctataatgcgcctccatcgaca",
    b"aaacaatttcagaatagacaaaaactctgagtgtaataatgtagcctcgtgtcttgc",
    b"tctcaacgtaacactttacagcggcgcgtcatttgatatgatgcgccccgcttcccg",
    b"gcaaataatcaatgtggacttttctgccgtgattatagacacttttgttacgcgttt",
    b"gacaccatcgaatggcgcaaaacctttcgcggtatggcatgatagcgcccggaagag",
    b"aaaaacgtcatcgcttgcattagaaaggtttctggccgaccttataaccattaatta",
    b"tctgaaatgagctgttgacaattaatcatcgaactagttaactagtacgcaagttca",
    b"accggaagaaaaccgtgacattttaacacgtttgttacaaggtaaaggcgacgccgc",
    b"aaattaaaattttattgacttaggtcactaaatactttaaccaatataggcatagcg",
    b"ttgtcataatcgacttgtaaaccaaattgaaaagatttaggtttacaagtctacacc",
    b"catcctcgcaccagtcgacgacggtttacgctttacgtatagtggcgacaatttttt",
    b"tccagtataatttgttggcataattaagtacgacgagtaaaattacatacctgcccg",
    b"acagttatccactattcctgtggataaccatgtgtattagagttagaaaacacgagg",
    b"tgtgcagtttatggttccaaaatcgccttttgctgtatatactcacagcataactgt",
    b"ctgttgttcagtttttgagttgtgtataacccctcattctgatcccagcttatacgg",
    b"attacaaaaagtgctttctgaactgaacaaaaaagagtaaagttagtcgcgtagggt",
    b"atgcgcaacgcggggtgacaagggcgcgcaaaccctctatactgcgcgccgaagctg",
    b"taaaaaactaacagttgtcagcctgtcccgcttataagatcatacgccgttatacgt",
    b"atgcaattttttagttgcatgaactcgcatgtctccatagaatgcgcgctacttgat",
    b"ccttgaaaaagaggttgacgctgcaaggctctatacgcataatgcgccccgcaacgc",
    b"tcgttgtatatttcttgacaccttttcggcatcgccctaaaattcggcgtcctcata",
    b"ccgtttattttttctacccatatccttgaagcggtgttataatgccgcgccctcgat",
    b"ttcgcatatttttcttgcaaagttgggttgagctggctagattagccagccaatctt",
    b"tgtaaactaatgcctttacgtgggcggtgattttgtctacaatcttacccccacgta",
    b"gatcgcacgatctgtatacttatttgagtaaattaacccacgatcccagccattctt",
    b"aacgcatacggtattttaccttcccagtcaagaaaacttatcttattcccacttttc",
    b"ttagcggatcctacctgacgctttttatcgcaactctctactgtttctccatacccg",
    b"gccttctccaaaacgtgttttttgttgttaattcggtgtagacttgtaaacctaaat",
    b"cagaaacgttttattcgaacatcgatctcgtcttgtgttagaattctaacatacggt",
    b"cactaatttattccatgtcacacttttcgcatctttgttatgctatggttatttcat",
    b"atataaaaaagttcttgctttctaacgtgaaagtggtttaggttaaaagacatcagt",
    b"caaggtagaatgctttgccttgtcggcctgattaatggcacgatagtcgcatcggat",
    b"ggccaaaaaatatcttgtactatttacaaaacctatggtaactctttaggcattcct",
    b"taggcaccccaggctttacactttatgcttccggctcgtatgttgtgtggaattgtg",
    b"ccatcaaaaaaatattctcaacataaaaaactttgtgtaatacttgtaacgctacat",
    b"tggggacgtcgttactgatccgcacgtttatgatatgctatcgtactctttagcgag",
    b"tcagaaatattatggtgatgaactgtttttttatccagtataatttgttggcataat",
];

const DATA_MINUS: [&[u8]; 53] = [
    b"atatgaacgttgagactgccgctgagttatcagctgtgaacgacattctggcgtcta",
    b"cgaacgagtcaatcagaccgctttgactctggtattactgtgaacattattcgtctc",
    b"caatggcctctaaacgggtcttgaggggttttttgctgaaaggaggaactatatgcg",
    b"ttgacctactacgccagcattttggcggtgtaagctaaccattccggttgactcaat",
    b"cgtctatcggtgaacctccggtatcaacgctggaaggtgacgctaacgcagatgcag",
    b"gccaatcaatcaagaacttgaagggtggtatcagccaacagcctgacatccttcgtt",
    b"tggatggacgttcaacattgaggaaggcataacgctactacctgatgtttactccaa",
    b"gaggtggctatgtgtatgaccgaacgagtcaatcagaccgctttgactctggtatta",
    b"cgtagcgcatcagtgctttcttactgtgagtacgcaccagcgccagaggacgacgac",
    b"cgaccgaagcgagcctcgtcctcaatggcctctaaacgggtcttgaggggttttttg",
    b"ctacggtgggtacaatatgctggatggagatgcgttcacttctggtctactgactcg",
    b"atagtctcagagtcttgacctactacgccagcattttggcggtgtaagctaaccatt",
    b"aactcaaggctgatacggcgagacttgcgagccttgtccttgcggtacacagcagcg",
    b"ttactgtgaacattattcgtctccgcgactacgatgagatgcctgagtgcttccgtt",
    b"tattctcaacaagattaaccgacagattcaatctcgtggatggacgttcaacattga",
    b"aacgagtcaatcagaccgctttgactctggtattactgtgaacattattcgtctccg",
    b"aagtgcttagcttcaaggtcacggatacgaccgaagcgagcctcgtcctcaatggcc",
    b"gaagaccacgcctcgccaccgagtagacccttagagagcatgtcagcctcgacaact",
    b"ttagagagcatgtcagcctcgacaacttgcataaatgctttcttgtagacgtgccct",
    b"tattcgtctccgcgactacgatgagatgcctgagtgcttccgttactggattgtcac",
    b"tgctgaaaggaggaactatatgcgctcatacgatatgaacgttgagactgccgctga",
    b"catgaactcaaggctgatacggcgagacttgcgagccttgtccttgcggtacacagc",
    b"ttcgtctccgcgactacgatgagatgcctgagtgcttccgttactggattgtcacca",
    b"catgtcagcctcgacaacttgcataaatgctttcttgtagacgtgccctacgcgctt",
    b"aggaggaactacgcaaggttggaacatcggagagatgccagccagcgcacctgcacg",
    b"tctcaacaagattaaccgacagattcaatctcgtggatggacgttcaacattgagga",
    b"tgaagtgcttagcttcaaggtcacggatacgaccgaagcgagcctcgtcctcaatgg",
    b"ctatatgcgctcatacgatatgaacgttgagactgccgctgagttatcagctgtgaa",
    b"gcggcagcacgtttccacgcggtgagagcctcaggattcatgtcgatgtcttccggt",
    b"atccctaatgtctacttccggtcaatccatctacgttaaccgaggtggctatgtgta",
    b"tggcgtctatcggtgaacctccggtatcaacgctggaaggtgacgctaacgcagatg",
    b"tctcgtggatggacgttcaacattgaggaaggcataacgctactacctgatgtttac",
    b"tattggcttgctcaagcatgaactcaaggctgatacggcgagacttgcgagccttgt",
    b"tagagggtgtactccaagaagaggaagatgaggctagacgtctctgcatggagtatg",
    b"cagcggcagcacgtttccacgcggtgagagcctcaggattcatgtcgatgtcttccg",
    b"ttacgttggcgaccgctaggactttcttgttgattttccatgcggtgttttgcgcaa",
    b"acgctaacgcagatgcagcgaacgctcggcgtattctcaacaagattaaccgacaga",
    b"ggtgttttgcgcaatgttaatcgctttgtacacctcaggcatgtaaacgtcttcgta",
    b"aaccattccggttgactcaatgagcatctcgatgcagcgtactcctacatgaataga",
    b"agacgtctctgcatggagtatgagatggactacggtgggtacaatatgctggatgga",
    b"tgttgattttccatgcggtgttttgcgcaatgttaatcgctttgtacacctcaggca",
    b"tgcacgggttgcgatagcctcagcgtattcaggtgcgagttcgatagtctcagagtc",
    b"aggcatgtaaacgtcttcgtagcgcatcagtgctttcttactgtgagtacgcaccag",
    b"ccgagtagacccttagagagcatgtcagcctcgacaacttgcataaatgctttcttg",
    b"cgctaggactttcttgttgattttccatgcggtgttttgcgcaatgttaatcgcttt",
    b"tatgaccgaacgagtcaatcagaccgctttgactctggtattactgtgaacattatt",
    b"agagggtgtactccaagaagaggaagatgaggctagacgtctctgcatggagtatga",
    b"gagagcatgtcagcctcgacaacttgcataaatgctttcttgtagacgtgccctacg",
    b"cctcaatggcctctaaacgggtcttgaggggttttttgctgaaaggaggaactatat",
    b"gtattctcaacaagattaaccgacagattcaatctcgtggatggacgttcaacattg",
    b"cgcgactacgatgagatgcctgagtgcttccgttactggattgtcaccaaggcttcc",
    b"ctcgtcctcaatggcctctaaacgggtcttgaggggttttttgctgaaaggaggaac",
    b"taacattaataaataaggaggctctaatggcactcattagccaatcaatcaagaact",
];

fn encode_sequence<H: HyperVector>(seq: &[u8], alphabet: &Alphabet<H>, n: usize) -> H {
    let mut acc = <H as HyperVector>::UnitAccumulator::default();

    // For each window, we recompute the n-gram hypervector from scratch.
    // this works for all kinds of HDVs
    // a potentially faster way for binary and bipolar is to 'unbind' the oldest symbol
    // for real and complex HDVs this doesn't work well because unbind is noisy
    let n0 = if n > 1 { n - 1 } else { 1 };
    let n1 = n + 1;
    for x in n0..=n1 {
        for window in seq.windows(x) {
            let mut ngram = H::ident();
            for &c in window {
                let sym = alphabet.get(c);
                ngram = ngram.permute(1).bind(sym);
            }
            acc.add(&ngram);
        }
    }

    acc.finalize()
}

fn train<H: HyperVector>(
    data_plus: &[&[u8]],
    data_minus: &[&[u8]],
    alphabet: &Alphabet<H>,
    n: usize,
) -> (H, H) {
    let mut acc = <H as HyperVector>::UnitAccumulator::default();
    for seq in data_plus {
        acc.add(&encode_sequence(seq, alphabet, n));
    }
    let plus = acc.finalize();

    let mut acc = <H as HyperVector>::UnitAccumulator::default();
    for seq in data_minus {
        acc.add(&encode_sequence(seq, alphabet, n));
    }
    (plus, acc.finalize())
}

fn run_loo<H: HyperVector>(alphabet: &Alphabet<H>, n: usize) -> (usize, usize) {
    let mut correct = 0;
    let mut total = 0;

    // PLUS samples
    for i in 0..DATA_PLUS.len() {
        let train_plus: Vec<_> = DATA_PLUS
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| *s)
            .collect();

        let (plus_hv, minus_hv) = train(&train_plus, &DATA_MINUS, &alphabet, n);

        let h = encode_sequence(DATA_PLUS[i], &alphabet, n);

        if h.distance(&plus_hv) < h.distance(&minus_hv) {
            correct += 1;
        }
        total += 1;
    }

    // MINUS samples
    for i in 0..DATA_MINUS.len() {
        let train_minus: Vec<_> = DATA_MINUS
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| *s)
            .collect();

        let (plus_hv, minus_hv) = train(&DATA_PLUS, &train_minus, &alphabet, n);

        let h = encode_sequence(DATA_MINUS[i], &alphabet, n);

        if h.distance(&plus_hv) > h.distance(&minus_hv) {
            correct += 1;
        }
        total += 1;
    }

    (correct, total)
}

fn run_suite<H: HyperVector>(alphabet: &Alphabet<H>, args: &Args) {
    for n in 2..=args.ngram {
        let (correct, total) = run_loo(alphabet, n);
        let acc = 100.0 * correct as f64 / total as f64;
        println!("{n}-gram: Accuracy: {acc:.2}% ({correct}/{total})");
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "modular"])]
    mode: String,

    #[arg(long, default_value_t = 1024)]
    /// one of 1024, 2048, 4096, 8192, 10048, 20096, 40192
    dim: usize,

    #[arg(long, default_value_t = 3)]
    ngram: usize,
}

fn main() {
    hdv!(binary, Bin1024, 1024);
    hdv!(binary, Bin2048, 2048);
    hdv!(binary, Bin4096, 4096);
    hdv!(binary, Bin8192, 8192);
    hdv!(binary, Bin10048, 10048);
    hdv!(binary, Bin20096, 20096);
    hdv!(binary, Bin40192, 40192);
    hdv!(modular, Mod1024, 1024);
    hdv!(modular, Mod2048, 2048);
    hdv!(modular, Mod5024, 5024);

    let args = Args::parse();
    println!(
        "Mode: {} N-gram: {} Dim: {}",
        args.mode, args.ngram, args.dim
    );

    let mut rng = MersenneTwister64::new(42);

    match (args.mode.as_str(), args.dim) {
        ("binary", 1024) => run_suite(&Alphabet::<Bin1024>::new(&mut rng), &args),
        ("binary", 2048) => run_suite(&Alphabet::<Bin2048>::new(&mut rng), &args),
        ("binary", 4096) => run_suite(&Alphabet::<Bin4096>::new(&mut rng), &args),
        ("binary", 8192) => run_suite(&Alphabet::<Bin8192>::new(&mut rng), &args),
        ("binary", 10048) => run_suite(&Alphabet::<Bin10048>::new(&mut rng), &args),
        ("binary", 20096) => run_suite(&Alphabet::<Bin20096>::new(&mut rng), &args),
        ("binary", 40192) => run_suite(&Alphabet::<Bin40192>::new(&mut rng), &args),
        ("modular", 1024) => run_suite(&Alphabet::<Mod1024>::new(&mut rng), &args),
        ("modular", 2048) => run_suite(&Alphabet::<Mod2048>::new(&mut rng), &args),
        ("modular", 5024) => run_suite(&Alphabet::<Mod5024>::new(&mut rng), &args),
        _ => {
            eprintln!("Unsupported combination: {args:?}");
            std::process::exit(1);
        }
    };
}
