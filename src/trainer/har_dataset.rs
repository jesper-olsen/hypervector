use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

// ls -l UCI\ HAR\ Dataset/test
//
//        352 Nov 30  2012 Inertial Signals
//       7934 Nov 30  2012 subject_test.txt
//   26458166 Nov 30  2012 X_test.txt
//       5894 Nov 30  2012 y_test.txt
//
// ls -l UCI\ HAR\ Dataset/train
//
//        352 Nov 30  2012 Inertial Signals
//      20152 Nov 30  2012 subject_train.txt
//   66006256 Nov 30  2012 X_train.txt
//      14704 Nov 30  2012 y_train.txt

// subject_test.txt and subject_train.txt
// subject id (1..30) one per line, subject on line n refers to feature vector on line n in X_test/X_train/y_test/y_train
//
// y_test.txt and y_train.txt
// activity (1-6), line n refers tofeature v
// activity (1-6) one per line, activity on line n refers to feature vector on line n in X_test/X_train/y_test/y_train

pub const NUM_CLASSES: usize = 6;
pub const N_FEATURES: usize = 561;
pub type Sample = [f32; N_FEATURES];

pub struct Dataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
    pub train_labels: Vec<Label>,
    pub test_labels: Vec<Label>,
    pub train_subjects: Vec<u8>,
    pub test_subjects: Vec<u8>,
}

impl Dataset {
    pub fn load(dir: &str) -> io::Result<Self> {
        let base = Path::new(dir);
        Ok(Self {
            train: load_features(&base.join("train/X_train.txt"))?,
            test: load_features(&base.join("test/X_test.txt"))?,
            train_labels: load_labels(&base.join("train/y_train.txt"))?,
            test_labels: load_labels(&base.join("test/y_test.txt"))?,
            train_subjects: load_subjects(&base.join("train/subject_train.txt"))?,
            test_subjects: load_subjects(&base.join("test/subject_test.txt"))?,
        })
    }
}

fn load_features(path: &Path) -> io::Result<Vec<Sample>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line?;
            let mut sample = [0.0f32; N_FEATURES];
            let mut tokens = line.split_whitespace();

            for (i, slot) in sample.iter_mut().enumerate() {
                let tok = tokens.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("expected {N_FEATURES} features, got {i}"),
                    )
                })?;
                *slot = f32::from_str(tok)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            }

            if tokens.next().is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected {N_FEATURES} features, got more"),
                ));
            }
            Ok(sample)
        })
        .collect()
}

fn load_labels(path: &Path) -> io::Result<Vec<Label>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            let n: u8 = line?
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Label::from_label(n))
        })
        .collect()
}

fn load_subjects(path: &Path) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    io::BufReader::new(file)
        .lines()
        .map(|line| {
            line?
                .trim()
                .parse::<u8>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Label {
    Walking = 0,
    WalkingUpstairs = 1,
    WalkingDownstairs = 2,
    Sitting = 3,
    Standing = 4,
    Laying = 5,
}

impl Label {
    pub fn from_label(n: u8) -> Self {
        match n {
            1 => Label::Walking,
            2 => Label::WalkingUpstairs,
            3 => Label::WalkingDownstairs,
            4 => Label::Sitting,
            5 => Label::Standing,
            6 => Label::Laying,
            _ => panic!("Invalid activity label: {n}"),
        }
    }
}

impl From<Label> for usize {
    fn from(a: Label) -> usize {
        a as usize
    }
}
