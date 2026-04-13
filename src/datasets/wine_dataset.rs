use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

pub const NUM_CLASSES: usize = 2;
pub const N_FEATURES: usize = 11;
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
        let fname1 = "winequality_train.csv";
        let fname2 = "winequality_test.csv";
        let (train, train_labels) = load_samples(&base.join(fname1))?;
        let (test, test_labels) = load_samples(&base.join(fname2))?;
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

    for line in io::BufReader::new(file).lines().skip(1) {
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

        let label_f: u8 = u8::from_str(tokens[N_FEATURES])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let label = Label::from_label(label_f);

        samples.push(sample);
        labels.push(label);
    }

    Ok((samples, labels))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Label {
    Bad = 0,
    Good = 1,
}

impl Label {
    pub fn from_label(n: u8) -> Self {
        match n {
            0 => Label::Bad,
            1 => Label::Good,
            _ => panic!("Invalid class label: {n}"),
        }
    }
}

impl From<Label> for usize {
    fn from(a: Label) -> usize {
        a as usize
    }
}
