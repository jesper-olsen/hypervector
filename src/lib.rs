use std::fs::File;
use std::io::{BufWriter, Read, Write};

pub mod datasets;
pub mod encoding;
pub mod trainer;
pub mod types;

// Re-exports
pub use types::traits::{Accumulator, HyperVector, UnitAccumulator};

pub fn save_hypervectors_to_csv<H: HyperVector>(
    filename: &str,
    vectors: &[H],
) -> Result<(), std::io::Error> {
    let mut writer = BufWriter::new(File::create(filename)?);

    for v in vectors {
        let values = v.unpack();
        for (i, val) in values.iter().enumerate() {
            write!(writer, "{val}")?;
            if i < values.len() - 1 {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }
    Ok(())
}

pub fn write_hypervectors<H: HyperVector>(vec: &[H], mut file: File) -> std::io::Result<()> {
    // Write number of HDVs
    let len = vec.len();
    file.write_all(&len.to_le_bytes())?;
    for hdv in vec {
        hdv.write(&mut file)?;
    }
    file.flush()
}

pub fn read_hypervectors<H: HyperVector>(mut file: File) -> std::io::Result<Vec<H>> {
    // Read number of HDVs
    let mut len_buf = [0u8; usize::BITS as usize / 8];
    file.read_exact(&mut len_buf)?;
    let len = usize::from_le_bytes(len_buf);
    println!("read_hypervectors: reading {len} HDVs");
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(H::read(&mut file)?);
    }
    Ok(vec)
}

pub fn cleanup<'a, T: HyperVector>(query: &T, vocab: &'a [(&str, T)]) -> &'a str {
    let mut best_label = vocab[0].0;
    let mut min_dist = query.distance(&vocab[0].1);

    for (label, v) in vocab.iter().skip(1) {
        let d = query.distance(v);
        if d < min_dist {
            min_dist = d;
            best_label = label;
        }
    }
    best_label
}

pub fn nearest<T: HyperVector>(query: &T, candidates: &[T]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut min_dist = query.distance(&candidates[0]);
    for (idx, v) in candidates.iter().enumerate().skip(1) {
        let d = query.distance(v);
        if d < min_dist {
            min_dist = d;
            best_idx = idx;
        }
    }
    (best_idx, min_dist)
}

pub fn nearest_two<T: HyperVector>(query: &T, candidates: &[T]) -> ((usize, f32), (usize, f32)) {
    assert!(candidates.len() >= 2);
    let mut first = (0, f32::MAX);
    let mut second = (0, f32::MAX);
    for (idx, v) in candidates.iter().enumerate() {
        let d = query.distance(v);
        if d < first.1 {
            second = first;
            first = (idx, d);
        } else if d < second.1 {
            second = (idx, d);
        }
    }
    (first, second)
}
