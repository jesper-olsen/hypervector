// Copyright (c) 2024 Jesper Olsen

use hypervector::{Accumulator, HyperVector, binary_hdv::BinaryHDV, bipolar_hdv::BipolarHDV};
use std::collections::hash_map::HashMap;
use std::collections::vec_deque::VecDeque;
use std::fs::File;
use std::io::{self, BufRead};

const LANG_MAP: [(&str, &str); 22] = [
    ("af", "afr"),
    ("bg", "bul"),
    ("cs", "ces"),
    ("da", "dan"),
    ("de", "deu"),
    ("el", "ell"),
    ("en", "eng"),
    ("es", "spa"),
    ("et", "est"),
    ("fi", "fin"),
    ("fr", "fra"),
    ("hu", "hun"),
    ("it", "ita"),
    ("lt", "lit"),
    ("lv", "lav"),
    ("nl", "nld"),
    ("pl", "pol"),
    ("pt", "por"),
    ("ro", "ron"),
    ("sk", "slk"),
    ("sl", "slv"),
    ("sv", "swe"),
];

pub fn compute_sum_hv<T: HyperVector>(
    fname: &str,
    n: usize,
    symbols: &mut HashMap<char, T>,
) -> Result<T, io::Error> {
    let file = File::open(fname)?;
    let reader = io::BufReader::new(file);
    let mut acc = T::Accumulator::default();

    for line in reader.lines().flatten() {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < n {
            continue;
        }
        let mut ngram = T::ident();
        let mut block: Vec<char> = Vec::new();
        for &c in &chars[..n] {
            let b0 = symbols.entry(c).or_insert(T::new());
            block.insert(0, c);
            ngram = ngram.pmultiply(1, b0, 0);
        }
        for &c in &chars[n..] {
            let cn = block.pop().unwrap();
            let bn = symbols.get(&cn).unwrap();
            ngram = ngram.pmultiply(0, bn, n - 1); // forget
            let b0 = symbols.entry(c).or_insert(T::new());
            block.insert(0, c);
            ngram = ngram.pmultiply(1, b0, 0);
            acc.add(&ngram);
        }
    }
    Ok(acc.finalize())
}

// pub fn compute_sum_hv<T: HyperVector>(
//     fname: &str,
//     n: usize,
//     symbols: &mut HashMap<char, T>
// ) -> Result<T, io::Error> {
//     let file = File::open(fname)?;
//     let reader = io::BufReader::new(file);
//     let mut acc = T::Accumulator::default();

//     for line in reader.lines().flatten() {
//         let chars: Vec<char> = line.trim().chars().collect();
//         if chars.len() < n {
//             continue;
//         }

//         let mut block: VecDeque<char> = chars[..n].iter().copied().collect();
//         let mut ngram = T::ident();

//         for (i, &c) in block.iter().enumerate() {
//             let sym = symbols.entry(c).or_insert_with(T::new);
//             ngram = ngram.pmultiply(1, sym, i);
//         }

//         for &c in &chars[n..] {
//             let forget = block.pop_back().unwrap();
//             let forget_sym = symbols.get(&forget).unwrap();
//             ngram = ngram.pmultiply(0, forget_sym, n - 1);

//             let new_sym = symbols.entry(c).or_insert_with(T::new);
//             block.push_front(c);
//             ngram = ngram.pmultiply(1, new_sym, 0);

//             acc.add(&ngram);
//         }
//     }

//     Ok(acc.finalize())
// }

fn train<T: HyperVector>(
    n: usize,
) -> Result<(HashMap<char, T>, Vec<(&'static str, T)>), io::Error> {
    let mut symbols: HashMap<char, T> = HashMap::new();
    let mut languages: Vec<(&str, T)> = Vec::new();
    for (i, (_lxx, lxxx)) in LANG_MAP.iter().enumerate() {
        let fname = format!("LANG_ID/training_texts/{lxxx}.txt");
        println!("{i}/{}: Processing training file {fname}", LANG_MAP.len());
        let v = compute_sum_hv(&fname, n, &mut symbols)?;
        languages.push((lxxx, v));
    }
    Ok((symbols, languages))
}

fn test<T: HyperVector>(
    symbols: &mut HashMap<char, T>,
    languages: &[(&str, T)],
    n: usize,
) -> Result<(), io::Error> {
    let mut total = 0;
    let mut correct = 0;

    for (i, (lxx, lxxx)) in LANG_MAP.iter().enumerate() {
        println!("{i}/{}: Processing {lxxx}", LANG_MAP.len());

        let pattern = format!("LANG_ID/testing_texts/{lxx}_*.txt");
        for fname in glob::glob(&pattern).expect("wrong glob pattern") {
            let fname = fname.unwrap();
            let v = compute_sum_hv(fname.to_str().unwrap(), n, symbols)?;
            let mut min_lang = 0;
            let b = &languages[0].1;
            let mut dmin = T::distance(&v, b);
            for (j, (_lang, b)) in languages.iter().enumerate().skip(1) {
                let d = T::distance(&v, b);
                if d < dmin {
                    dmin = d;
                    min_lang = j;
                }
            }
            if &languages[min_lang].0 == lxxx {
                correct += 1;
            }
            total += 1;
        }
        if total > 0 {
            println!("+{} {lxxx}: Accuracy: {correct}/{total}={}", i + 1, {
                correct as f64 / total as f64
            })
        }
    }
    Ok(())
}

fn main() -> Result<(), io::Error> {
    let n = 3;
    println!("N-gram: {n}");

    let (mut symbols, languages) = train::<BipolarHDV<10048>>(n)?;
    test(&mut symbols, &languages, n)?;

    Ok(())
}
