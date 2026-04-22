use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::io::Write;

use clap::{Parser, ValueEnum};

use hypervector::encoding::{CategoricalEncoder, ScalarEncoder};
use hypervector::trainer::{
    Classifier, Trainer, ensemble_accuracy, lvq::LvqTrainer,
    multi_perceptron::PerceptronMultiTrainer, pa::PaTrainer, pa::PaVariant,
    perceptron::PerceptronTrainer,
};
use hypervector::types::{binary::Binary, complex::ComplexHDV, modular::Modular, real::RealHDV};
use hypervector::{HyperVector, UnitAccumulator, hdv};
use mersenne_twister_rs::MersenneTwister64;

// .csv file - 1st 3 lines

// age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income
// 39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
// 50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K

// ranges - mix of numeric and categorical feature columns
// 0: age 17-90
// 1: workclass
//    ?
//    Federal-gov
//    Local-gov
//    Never-worked
//    Private
//    Self-emp-inc
//    Self-emp-not-inc
//    State-gov
//    Without-pay
// 2 fnlwgt 12285-1490400
// 3: education
//    Assoc-acdm
//    Assoc-voc
//    Bachelors
//    Doctorate
//    HS-grad
//    Masters
//    Preschool
//    Prof-school
//    Some-college
//    1st-4th
//    5th-6th
//    7th-8th
//    9th
//    10th
//    11th
//    12th
// 4: education-num 1-16
// 5: maritial-status
//    Divorced
//    Married-AF-spouse
//    Married-civ-spouse
//    Married-spouse-absent
//    Never-married
//    Separated
//    Widowed
// 6: occupation
//    ?
//    Adm-clerical
//    Armed-Forces
//    Craft-repair
//    Exec-managerial
//    Farming-fishing
//    Handlers-cleaners
//    Machine-op-inspct
//    Other-service
//    Priv-house-serv
//    Prof-specialty
//    Protective-serv
//    Sales
//    Tech-support
//    Transport-moving
// 7: relationship
//    Husband
//    Not-in-family
//    Other-relative
//    Own-child
//    Unmarried
//    Wife
// 8: race
//    Amer-Indian-Eskimo
//    Asian-Pac-Islander
//    Black
//    Other
//    White
// 9: sex
//    Female
//    Male
// 10: capital-gain 0-99999
// 11: capital-loss 0-4356
// 12: hours-per-week 1-99
// 13: native-country
//     ?
//     Cambodia
//     Canada
//     China
//     Columbia
//     Cuba
//     Dominican-Republic
//     Ecuador
//     El-Salvador
//     England
//     France
//     Germany
//     Greece
//     Guatemala
//     Haiti
//     Holand-Netherlands
//     Honduras
//     Hong
//     Hungary
//     India
//     Iran
//     Ireland
//     Italy
//     Jamaica
//     Japan
//     Laos
//     Mexico
//     native-country
//     Nicaragua
//     Outlying-US(Guam-USVI-etc)
//     Peru
//     Philippines
//     Poland
//     Portugal
//     Puerto-Rico
//     Scotland
//     South
//     Taiwan
//     Thailand
//     Trinadad&Tobago
//     United-States
//     Vietnam
//     Yugoslavia
// 14: income
//     <=50K
//     >50K

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

pub const NUM_CLASSES: usize = 2;

// ── Categorical feature enums ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Workclass {
    Missing,
    FederalGov,
    LocalGov,
    NeverWorked,
    Private,
    SelfEmpInc,
    SelfEmpNotInc,
    StateGov,
    WithoutPay,
}

impl Workclass {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "?" => Workclass::Missing,
            "Federal-gov" => Workclass::FederalGov,
            "Local-gov" => Workclass::LocalGov,
            "Never-worked" => Workclass::NeverWorked,
            "Private" => Workclass::Private,
            "Self-emp-inc" => Workclass::SelfEmpInc,
            "Self-emp-not-inc" => Workclass::SelfEmpNotInc,
            "State-gov" => Workclass::StateGov,
            "Without-pay" => Workclass::WithoutPay,
            _ => return Err(invalid(format!("unknown workclass: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Education {
    AssocAcdm,
    AssocVoc,
    Bachelors,
    Doctorate,
    HsGrad,
    Masters,
    Preschool,
    ProfSchool,
    SomeCollege,
    First4th,
    Fifth6th,
    Seventh8th,
    Ninth,
    Tenth,
    Eleventh,
    Twelfth,
}

impl Education {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "Assoc-acdm" => Education::AssocAcdm,
            "Assoc-voc" => Education::AssocVoc,
            "Bachelors" => Education::Bachelors,
            "Doctorate" => Education::Doctorate,
            "HS-grad" => Education::HsGrad,
            "Masters" => Education::Masters,
            "Preschool" => Education::Preschool,
            "Prof-school" => Education::ProfSchool,
            "Some-college" => Education::SomeCollege,
            "1st-4th" => Education::First4th,
            "5th-6th" => Education::Fifth6th,
            "7th-8th" => Education::Seventh8th,
            "9th" => Education::Ninth,
            "10th" => Education::Tenth,
            "11th" => Education::Eleventh,
            "12th" => Education::Twelfth,
            _ => return Err(invalid(format!("unknown education: {s}"))),
        })
    }

    //#[inline]
    //fn idx(self) -> usize {
    //    self as usize
    //}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaritalStatus {
    Divorced,
    MarriedAfSpouse,
    MarriedCivSpouse,
    MarriedSpouseAbsent,
    NeverMarried,
    Separated,
    Widowed,
}

impl MaritalStatus {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "Divorced" => MaritalStatus::Divorced,
            "Married-AF-spouse" => MaritalStatus::MarriedAfSpouse,
            "Married-civ-spouse" => MaritalStatus::MarriedCivSpouse,
            "Married-spouse-absent" => MaritalStatus::MarriedSpouseAbsent,
            "Never-married" => MaritalStatus::NeverMarried,
            "Separated" => MaritalStatus::Separated,
            "Widowed" => MaritalStatus::Widowed,
            _ => return Err(invalid(format!("unknown marital-status: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Occupation {
    Missing,
    AdmClerical,
    ArmedForces,
    CraftRepair,
    ExecManagerial,
    FarmingFishing,
    HandlersCleaners,
    MachineOpInspct,
    OtherService,
    PrivHouseServ,
    ProfSpecialty,
    ProtectiveServ,
    Sales,
    TechSupport,
    TransportMoving,
}

impl Occupation {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "?" => Occupation::Missing,
            "Adm-clerical" => Occupation::AdmClerical,
            "Armed-Forces" => Occupation::ArmedForces,
            "Craft-repair" => Occupation::CraftRepair,
            "Exec-managerial" => Occupation::ExecManagerial,
            "Farming-fishing" => Occupation::FarmingFishing,
            "Handlers-cleaners" => Occupation::HandlersCleaners,
            "Machine-op-inspct" => Occupation::MachineOpInspct,
            "Other-service" => Occupation::OtherService,
            "Priv-house-serv" => Occupation::PrivHouseServ,
            "Prof-specialty" => Occupation::ProfSpecialty,
            "Protective-serv" => Occupation::ProtectiveServ,
            "Sales" => Occupation::Sales,
            "Tech-support" => Occupation::TechSupport,
            "Transport-moving" => Occupation::TransportMoving,
            _ => return Err(invalid(format!("unknown occupation: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relationship {
    Husband,
    NotInFamily,
    OtherRelative,
    OwnChild,
    Unmarried,
    Wife,
}

impl Relationship {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "Husband" => Relationship::Husband,
            "Not-in-family" => Relationship::NotInFamily,
            "Other-relative" => Relationship::OtherRelative,
            "Own-child" => Relationship::OwnChild,
            "Unmarried" => Relationship::Unmarried,
            "Wife" => Relationship::Wife,
            _ => return Err(invalid(format!("unknown relationship: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Race {
    AmerIndianEskimo,
    AsianPacIslander,
    Black,
    Other,
    White,
}

impl Race {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "Amer-Indian-Eskimo" => Race::AmerIndianEskimo,
            "Asian-Pac-Islander" => Race::AsianPacIslander,
            "Black" => Race::Black,
            "Other" => Race::Other,
            "White" => Race::White,
            _ => return Err(invalid(format!("unknown race: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sex {
    Female,
    Male,
}

impl Sex {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "Female" => Sex::Female,
            "Male" => Sex::Male,
            _ => return Err(invalid(format!("unknown sex: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeCountry {
    Missing,
    Cambodia,
    Canada,
    China,
    Columbia,
    Cuba,
    DominicanRepublic,
    Ecuador,
    ElSalvador,
    England,
    France,
    Germany,
    Greece,
    Guatemala,
    Haiti,
    HolandNetherlands,
    Honduras,
    Hong,
    Hungary,
    India,
    Iran,
    Ireland,
    Italy,
    Jamaica,
    Japan,
    Laos,
    Mexico,
    Nicaragua,
    OutlyingUS,
    Peru,
    Philippines,
    Poland,
    Portugal,
    PuertoRico,
    Scotland,
    South,
    Taiwan,
    Thailand,
    TrinadadTobago,
    UnitedStates,
    Vietnam,
    Yugoslavia,
}

impl NativeCountry {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "?" => NativeCountry::Missing,
            "Cambodia" => NativeCountry::Cambodia,
            "Canada" => NativeCountry::Canada,
            "China" => NativeCountry::China,
            "Columbia" => NativeCountry::Columbia,
            "Cuba" => NativeCountry::Cuba,
            "Dominican-Republic" => NativeCountry::DominicanRepublic,
            "Ecuador" => NativeCountry::Ecuador,
            "El-Salvador" => NativeCountry::ElSalvador,
            "England" => NativeCountry::England,
            "France" => NativeCountry::France,
            "Germany" => NativeCountry::Germany,
            "Greece" => NativeCountry::Greece,
            "Guatemala" => NativeCountry::Guatemala,
            "Haiti" => NativeCountry::Haiti,
            "Holand-Netherlands" => NativeCountry::HolandNetherlands,
            "Honduras" => NativeCountry::Honduras,
            "Hong" => NativeCountry::Hong,
            "Hungary" => NativeCountry::Hungary,
            "India" => NativeCountry::India,
            "Iran" => NativeCountry::Iran,
            "Ireland" => NativeCountry::Ireland,
            "Italy" => NativeCountry::Italy,
            "Jamaica" => NativeCountry::Jamaica,
            "Japan" => NativeCountry::Japan,
            "Laos" => NativeCountry::Laos,
            "Mexico" => NativeCountry::Mexico,
            "Nicaragua" => NativeCountry::Nicaragua,
            "Outlying-US(Guam-USVI-etc)" => NativeCountry::OutlyingUS,
            "Peru" => NativeCountry::Peru,
            "Philippines" => NativeCountry::Philippines,
            "Poland" => NativeCountry::Poland,
            "Portugal" => NativeCountry::Portugal,
            "Puerto-Rico" => NativeCountry::PuertoRico,
            "Scotland" => NativeCountry::Scotland,
            "South" => NativeCountry::South,
            "Taiwan" => NativeCountry::Taiwan,
            "Thailand" => NativeCountry::Thailand,
            "Trinadad&Tobago" => NativeCountry::TrinadadTobago,
            "United-States" => NativeCountry::UnitedStates,
            "Vietnam" => NativeCountry::Vietnam,
            "Yugoslavia" => NativeCountry::Yugoslavia,
            _ => return Err(invalid(format!("unknown native-country: {s}"))),
        })
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

// ── Label ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Label {
    AtMost50K = 0,
    Above50K = 1,
}

impl Label {
    fn parse(s: &str) -> io::Result<Self> {
        Ok(match s {
            "<=50K" | "<=50K." => Label::AtMost50K,
            ">50K" | ">50K." => Label::Above50K,
            _ => return Err(invalid(format!("unknown income label: {s}"))),
        })
    }

    //#[inline]
    //fn idx(self) -> usize {
    //    self as usize
    //}
}

impl From<Label> for usize {
    fn from(l: Label) -> usize {
        l as usize
    }
}

// ── Sample ────────────────────────────────────────────────────────────────────
//
// Numeric columns are stored as raw f32; normalization (min-max, z-score, etc.)
// is left to the hypervector encoding layer so it can apply dataset-wide stats.
//
// Numeric ranges for reference:
//   age            17 – 90
//   fnlwgt      12285 – 1 490 400
//   education_num   1 – 16
//   capital_gain    0 – 99 999
//   capital_loss    0 – 4 356
//   hours_per_week  1 – 99

#[derive(Debug, Clone)]
pub struct Sample {
    // numeric
    pub age: f32,
    pub fnlwgt: f32,
    pub education_num: f32,
    pub capital_gain: f32,
    pub capital_loss: f32,
    pub hours_per_week: f32,
    // categorical
    pub workclass: Workclass,
    pub education: Education,
    pub marital_status: MaritalStatus,
    pub occupation: Occupation,
    pub relationship: Relationship,
    pub race: Race,
    pub sex: Sex,
    pub native_country: NativeCountry,
}

// ── Dataset ───────────────────────────────────────────────────────────────────

pub struct Dataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
    pub train_labels: Vec<Label>,
    pub test_labels: Vec<Label>,
}

impl Dataset {
    /// Load from a directory that contains `adult_train.csv` and `adult_test.csv`.
    pub fn load(dir: &str) -> io::Result<Self> {
        let base = Path::new(dir);
        let (train, train_labels) = load_samples(&base.join("adult_train.csv"))?;
        let (test, test_labels) = load_samples(&base.join("adult_test.csv"))?;
        Ok(Self {
            train,
            test,
            train_labels,
            test_labels,
        })
    }
}

// ── Parsing ───────────────────────────────────────────────────────────────────

fn load_samples(path: &Path) -> io::Result<(Vec<Sample>, Vec<Label>)> {
    let file = fs::File::open(path)?;
    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for (line_no, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip blank lines and the header (first line, or lines starting with "age")
        if line.is_empty() || line.starts_with("age") {
            continue;
        }

        let tokens: Vec<&str> = line.split(',').map(str::trim).collect();

        // Expect exactly 15 tokens: 14 features + 1 label
        if tokens.len() != 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {line_no}: expected 15 tokens, got {}", tokens.len()),
            ));
        }

        let sample = parse_sample(&tokens, line_no)?;
        let label = Label::parse(tokens[14]).map_err(|e| ctx(e, line_no))?;

        samples.push(sample);
        labels.push(label);
    }

    Ok((samples, labels))
}

fn parse_sample(t: &[&str], ln: usize) -> io::Result<Sample> {
    let parse_f32 = |s: &str, col: &str| -> io::Result<f32> {
        f32::from_str(s).map_err(|e| {
            ctx(
                io::Error::new(io::ErrorKind::InvalidData, format!("column {col}: {e}")),
                ln,
            )
        })
    };

    Ok(Sample {
        age: parse_f32(t[0], "age")?,
        workclass: Workclass::parse(t[1]).map_err(|e| ctx(e, ln))?,
        fnlwgt: parse_f32(t[2], "fnlwgt")?,
        education: Education::parse(t[3]).map_err(|e| ctx(e, ln))?,
        education_num: parse_f32(t[4], "education-num")?,
        marital_status: MaritalStatus::parse(t[5]).map_err(|e| ctx(e, ln))?,
        occupation: Occupation::parse(t[6]).map_err(|e| ctx(e, ln))?,
        relationship: Relationship::parse(t[7]).map_err(|e| ctx(e, ln))?,
        race: Race::parse(t[8]).map_err(|e| ctx(e, ln))?,
        sex: Sex::parse(t[9]).map_err(|e| ctx(e, ln))?,
        capital_gain: parse_f32(t[10], "capital-gain")?,
        capital_loss: parse_f32(t[11], "capital-loss")?,
        hours_per_week: parse_f32(t[12], "hours-per-week")?,
        native_country: NativeCountry::parse(t[13]).map_err(|e| ctx(e, ln))?,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn ctx(e: io::Error, line_no: usize) -> io::Error {
    io::Error::new(e.kind(), format!("line {line_no}: {e}"))
}

// ── AdultEncoder ──────────────────────────────────────────────────────────────

pub struct AdultEncoder<H: HyperVector> {
    // numeric
    age: ScalarEncoder<H>,
    //fnlwgt: ScalarEncoder<H>,
    education_num: ScalarEncoder<H>,
    capital_gain: ScalarEncoder<H>,
    capital_loss: ScalarEncoder<H>,
    hours_per_week: ScalarEncoder<H>,

    // categorical
    workclass: CategoricalEncoder<H>,
    //education: CategoricalEncoder<H>,
    marital_status: CategoricalEncoder<H>,
    occupation: CategoricalEncoder<H>,
    relationship: CategoricalEncoder<H>,
    race: CategoricalEncoder<H>,
    sex: CategoricalEncoder<H>,
    native_country: CategoricalEncoder<H>,

    // field keys
    keys: [H; 12],
}

impl<H: HyperVector> AdultEncoder<H> {
    const IDX_AGE: usize = 0;
    //const IDX_FNLWGT: usize = 0;
    const IDX_EDUCATION_NUM: usize = 1;
    const IDX_CAPITAL_GAIN: usize = 2;
    const IDX_CAPITAL_LOSS: usize = 3;
    const IDX_HOURS_PER_WEEK: usize = 4;
    const IDX_WORKCLASS: usize = 5;
    //const IDX_EDUCATION: usize = 6;
    const IDX_MARITIAL_STATUS: usize = 6;
    const IDX_OCCUPATION: usize = 7;
    const IDX_RELATIONSHIP: usize = 8;
    const IDX_RACE: usize = 9;
    const IDX_SEX: usize = 10;
    const IDX_NATIVE_COUNTRY: usize = 11;

    pub fn new(rng: &mut impl Rng) -> Self {
        Self {
            // numeric ranges (your comments already define these)
            age: ScalarEncoder::new(17.0, 90.0, 128, rng),
            //fnlwgt: ScalarEncoder::new(12285.0, 1_490_400.0, 128, rng),
            education_num: ScalarEncoder::new(1.0, 16.0, 16, rng),
            //capital_gain: ScalarEncoder::new(0.0, 100_000.0, 64, rng),
            //capital_loss: ScalarEncoder::new(0.0, 5000.0, 64, rng),
            capital_gain: ScalarEncoder::new(0.0, 12.0, 64, rng),
            capital_loss: ScalarEncoder::new(0.0, 9.0, 64, rng),
            hours_per_week: ScalarEncoder::new(1.0, 99.0, 128, rng),

            // categorical sizes = enum cardinality
            workclass: CategoricalEncoder::new(9, rng),
            //education: CategoricalEncoder::new(16, rng),
            marital_status: CategoricalEncoder::new(7, rng),
            occupation: CategoricalEncoder::new(15, rng),
            relationship: CategoricalEncoder::new(6, rng),
            race: CategoricalEncoder::new(5, rng),
            sex: CategoricalEncoder::new(2, rng),
            native_country: CategoricalEncoder::new(42, rng),

            //keys: (0..14).map(|_| H::random(rng)).collect(),
            keys: std::array::from_fn(|_| H::random(rng)),
        }
    }

    pub fn encode(&self, s: &Sample) -> H {
        let mut acc = H::UnitAccumulator::new();

        // numeric
        acc.add(&self.age.encode(s.age).bind(&self.keys[Self::IDX_AGE]));
        //acc.add(&self.fnlwgt.encode(s.fnlwgt).bind(&self.keys[Self::IDX_FNLWGT]));
        acc.add(
            &self
                .education_num
                .encode(s.education_num)
                .bind(&self.keys[Self::IDX_EDUCATION_NUM]),
        );
        let cg = (s.capital_gain + 1.0).ln(); // log1p: 0 stays 0, range becomes ~0-11.5
        let cl = (s.capital_loss + 1.0).ln(); // range becomes ~0-8.4
        acc.add(
            &self
                .capital_gain
                .encode(cg)
                .bind(&self.keys[Self::IDX_CAPITAL_GAIN]),
        );
        acc.add(
            &self
                .capital_loss
                .encode(cl)
                .bind(&self.keys[Self::IDX_CAPITAL_LOSS]),
        );

        //acc.add(
        //    &self
        //        .capital_gain
        //        .encode(s.capital_gain)
        //        .bind(&self.keys[Self::IDX_CAPITAL_GAIN]),
        //);
        //acc.add(
        //    &self
        //        .capital_loss
        //        .encode(s.capital_loss)
        //        .bind(&self.keys[Self::IDX_CAPITAL_LOSS]),
        //);
        acc.add(
            &self
                .hours_per_week
                .encode(s.hours_per_week)
                .bind(&self.keys[Self::IDX_HOURS_PER_WEEK]),
        );

        // categorical
        acc.add(
            &self
                .workclass
                .encode(s.workclass.idx())
                .bind(&self.keys[Self::IDX_WORKCLASS]),
        );
        //acc.add(
        //    &self
        //        .education
        //        .encode(s.education.idx())
        //        .bind(&self.keys[Self::IDX_EDUCATION]),
        //);
        acc.add(
            &self
                .marital_status
                .encode(s.marital_status.idx())
                .bind(&self.keys[Self::IDX_MARITIAL_STATUS]),
        );
        acc.add(
            &self
                .occupation
                .encode(s.occupation.idx())
                .bind(&self.keys[Self::IDX_OCCUPATION]),
        );
        acc.add(
            &self
                .relationship
                .encode(s.relationship.idx())
                .bind(&self.keys[Self::IDX_RELATIONSHIP]),
        );
        acc.add(
            &self
                .race
                .encode(s.race.idx())
                .bind(&self.keys[Self::IDX_RACE]),
        );
        acc.add(&self.sex.encode(s.sex.idx()).bind(&self.keys[Self::IDX_SEX]));
        acc.add(
            &self
                .native_country
                .encode(s.native_country.idx())
                .bind(&self.keys[Self::IDX_NATIVE_COUNTRY]),
        );

        acc.finalize()
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TrainerKind {
    Perceptron,
    Pa,
    Pai,
    Paii,
    Multi,
    Lvq,
}

impl fmt::Display for TrainerKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainerKind::Perceptron => write!(f, "perceptron"),
            TrainerKind::Pa => write!(f, "pa"),
            TrainerKind::Pai => write!(f, "pai"),
            TrainerKind::Paii => write!(f, "paii"),
            TrainerKind::Multi => write!(f, "multi"),
            TrainerKind::Lvq => write!(f, "lvq"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "binary", value_parser=["binary", "bipolar", "real", "complex", "modular"])]
    mode: String,

    #[arg(long, default_value_t = 8192, value_parser = valid_dim)]
    /// One of 1024, 2048, 4096, 8192, 16384
    dim: usize,

    #[arg(long, default_value_t = TrainerKind::Perceptron)]
    trainer: TrainerKind,

    #[arg(long, default_value_t = 1)]
    /// number of prototypes per class
    prototypes: usize,

    #[arg(long, default_value_t = 0.25)]
    /// lvq window
    window: f32,

    #[arg(long, default_value_t = 1000)]
    epochs: usize,

    #[arg(long, default_value_t = 9)]
    ensemble_size: usize,
}

fn valid_dim(s: &str) -> Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("{s} is not a number"))?;
    match n {
        1024 | 2048 | 4096 | 8192 | 16384 => Ok(n),
        _ => Err(format!(
            "{n} is not a supported dimension (1024, 2048, 8192, 16384)"
        )),
    }
}

hdv!(binary, BinaryHDV1024, 1024);
hdv!(binary, BinaryHDV2048, 2048);
hdv!(binary, BinaryHDV4096, 4096);
hdv!(binary, BinaryHDV8192, 8192);
hdv!(binary, BinaryHDV16384, 16384);
hdv!(modular, ModularHDV1024, 1024);
hdv!(modular, ModularHDV2048, 2048);
hdv!(modular, ModularHDV4096, 4096);
hdv!(modular, ModularHDV8192, 8192);
hdv!(modular, ModularHDV16384, 16384);
hdv!(real, RealHDV1024, 1024);
hdv!(real, RealHDV2048, 2048);
hdv!(complex, ComplexHDV1024, 1024);

fn train<T, Tr>(mut trainer: Tr, epochs: usize) -> Tr::Model
where
    T: HyperVector,
    Tr: Trainer<T>,
{
    for epoch in 1..=epochs {
        let r = trainer.step(epoch);
        print!(
            "Epoch {epoch}: Training Accuracy {}/{}={:.2}%\r",
            r.correct,
            r.total(),
            r.accuracy() * 100.0
        );
        std::io::stdout().flush().unwrap();
        if r.errors == 0 {
            break;
        }
    }
    println!();
    trainer.into_model()
}

fn run<T: HyperVector + Sync + Send>(
    data: &Dataset,
    rng: &mut impl Rng,
    args: &Args,
) -> Vec<usize> {
    let encoder = AdultEncoder::<T>::new(rng);

    let train_hvs: Vec<T> = data
        .train
        .par_iter()
        .map(|sample| encoder.encode(sample))
        .collect();
    let test_hvs: Vec<T> = data
        .test
        .par_iter()
        .map(|sample| encoder.encode(sample))
        .collect();

    let k = args.prototypes;
    let epochs = args.epochs;

    match args.trainer {
        TrainerKind::Perceptron => {
            let trainer = PerceptronTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                None,
                &mut *rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Pa => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                None,
                PaVariant::Pa,
                &mut *rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Pai => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                None,
                PaVariant::PaI { c: 0.1 },
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Paii => {
            let trainer = PaTrainer::<T, Label, _, NUM_CLASSES>::new(
                &train_hvs,
                &data.train_labels,
                None,
                PaVariant::PaII { c: 1.0 },
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Multi => {
            let trainer = PerceptronMultiTrainer::<T, _>::new(
                &train_hvs,
                &data.train_labels,
                None,
                NUM_CLASSES,
                k,
                rng,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
        TrainerKind::Lvq => {
            let trainer = LvqTrainer::<T, _>::new(
                &train_hvs,
                &data.train_labels,
                None,
                NUM_CLASSES,
                k,
                rng,
                args.window,
            );
            train(trainer, epochs).classify_all(&test_hvs)
        }
    }
}

fn stats(v: &[f64]) -> Option<(f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }

    let mut min = v[0];
    let mut max = v[0];
    let mut sum = v[0];

    for &x in &v[1..] {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x;
    }

    let avg = sum / v.len() as f64;
    Some((100.0 * min, 100.0 * max, 100.0 * avg))
}

fn main() -> Result<(), io::Error> {
    let args = Args::parse();
    let ensemble_size = args.ensemble_size;
    let dir = "data/ADULT/";
    let Ok(dataset) = Dataset::load(dir) else {
        println!("Failed to load dataset: {dir}");
        return Ok(());
    };
    let mut rng = MersenneTwister64::default();

    let mut all_predictions = Vec::with_capacity(ensemble_size);
    let mut accs = Vec::with_capacity(ensemble_size);
    for i in 1..=ensemble_size {
        let preds = match (args.mode.as_str(), args.dim) {
            ("binary", 1024) => run::<BinaryHDV1024>(&dataset, &mut rng, &args),
            ("binary", 2048) => run::<BinaryHDV2048>(&dataset, &mut rng, &args),
            ("binary", 4096) => run::<BinaryHDV4096>(&dataset, &mut rng, &args),
            ("binary", 8192) => run::<BinaryHDV8192>(&dataset, &mut rng, &args),
            ("binary", 16384) => run::<BinaryHDV16384>(&dataset, &mut rng, &args),
            ("modular", 1024) => run::<ModularHDV1024>(&dataset, &mut rng, &args),
            ("modular", 2048) => run::<ModularHDV2048>(&dataset, &mut rng, &args),
            ("modular", 4096) => run::<ModularHDV4096>(&dataset, &mut rng, &args),
            ("modular", 8192) => run::<ModularHDV8192>(&dataset, &mut rng, &args),
            ("modular", 16384) => run::<ModularHDV16384>(&dataset, &mut rng, &args),
            ("real", 1024) => run::<RealHDV1024>(&dataset, &mut rng, &args),
            ("real", 2048) => run::<RealHDV2048>(&dataset, &mut rng, &args),
            ("complex", 1024) => run::<ComplexHDV1024>(&dataset, &mut rng, &args),
            _ => {
                eprintln!(
                    "Unsupported combination: mode={} dim={}",
                    args.mode, args.dim
                );
                return Ok(());
            }
        };
        let (correct, errors, acc) = ensemble_accuracy(
            std::slice::from_ref(&preds),
            &dataset.test_labels,
            NUM_CLASSES,
        );

        println!(
            "Model {i}/{ensemble_size} - test: {:.2}%  ({correct}/{})",
            acc * 100.0,
            correct + errors,
        );
        all_predictions.push(preds);
        accs.push(acc);
        if i > 2 {
            let (correct, errors, acc) =
                ensemble_accuracy(&all_predictions, &dataset.test_labels, NUM_CLASSES);
            println!(
                "Ensemble of {i} - test {:.2}%  ({correct}/{})",
                acc * 100.0,
                correct + errors,
            );
        }
        println!();
    }

    if let Some((min, max, avg)) = stats(&accs) {
        println!("Model accuracies - avg {avg:.2}%, min {min:.2}%, max {max:.2}")
    }

    Ok(())
}
