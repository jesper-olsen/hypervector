use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

pub const NUM_CLASSES: usize = 26;
pub const N_FEATURES: usize = 617;
pub type Sample = [f32; N_FEATURES];

pub struct Dataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
    pub train_labels: Vec<Label>,
    pub test_labels: Vec<Label>,
}

impl Dataset {
    pub fn load(dir: &str) -> io::Result<Self> {
        let base = Path::new(dir);
        let (train, train_labels) = load_samples(&base.join("isolet1+2+3+4.data"))?;
        let (test, test_labels) = load_samples(&base.join("isolet5.data"))?;
        Ok(Self {
            train,
            test,
            train_labels,
            test_labels,
        })
    }
}

fn load_samples(path: &Path) -> io::Result<(Vec<Sample>, Vec<Label>)> {
    let file = fs::File::open(path)?;
    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for line in io::BufReader::new(file).lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split(',').map(str::trim).collect();

        // Last token is the label, first N_FEATURES are the features
        if tokens.len() != N_FEATURES + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected {} tokens, got {}", N_FEATURES + 1, tokens.len()),
            ));
        }

        let mut sample = [0.0f32; N_FEATURES];
        for (i, tok) in tokens[..N_FEATURES].iter().enumerate() {
            sample[i] =
                f32::from_str(tok).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }

        // Label is a float like "1." or "26." in the file
        let label_f: f32 = f32::from_str(tokens[N_FEATURES])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let label = Label::from_label(label_f as u8);

        samples.push(sample);
        labels.push(label);
    }

    Ok((samples, labels))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Label {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
    I = 8,
    J = 9,
    K = 10,
    L = 11,
    M = 12,
    N = 13,
    O = 14,
    P = 15,
    Q = 16,
    R = 17,
    S = 18,
    T = 19,
    U = 20,
    V = 21,
    W = 22,
    X = 23,
    Y = 24,
    Z = 25,
}

impl Label {
    pub fn from_label(n: u8) -> Self {
        match n {
            1 => Label::A,
            2 => Label::B,
            3 => Label::C,
            4 => Label::D,
            5 => Label::E,
            6 => Label::F,
            7 => Label::G,
            8 => Label::H,
            9 => Label::I,
            10 => Label::J,
            11 => Label::K,
            12 => Label::L,
            13 => Label::M,
            14 => Label::N,
            15 => Label::O,
            16 => Label::P,
            17 => Label::Q,
            18 => Label::R,
            19 => Label::S,
            20 => Label::T,
            21 => Label::U,
            22 => Label::V,
            23 => Label::W,
            24 => Label::X,
            25 => Label::Y,
            26 => Label::Z,
            _ => panic!("Invalid class label: {n}"),
        }
    }
}

impl From<Label> for usize {
    fn from(a: Label) -> usize {
        a as usize
    }
}
