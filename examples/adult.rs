use rand::Rng;

use hypervector::encoding::{CategoricalEncoder, ScalarEncoder};
use hypervector::{HyperVector, UnitAccumulator, hdv};
use hypervector::types::binary::Binary;
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
            age: ScalarEncoder::new(17.0, 90.0, 64, rng),
            //fnlwgt: ScalarEncoder::new(12285.0, 1_490_400.0, 128, rng),
            education_num: ScalarEncoder::new(1.0, 16.0, 16, rng),
            capital_gain: ScalarEncoder::new(0.0, 100_000.0, 64, rng),
            capital_loss: ScalarEncoder::new(0.0, 5000.0, 64, rng),
            hours_per_week: ScalarEncoder::new(1.0, 99.0, 64, rng),

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
        acc.add(
            &self
                .capital_gain
                .encode(s.capital_gain)
                .bind(&self.keys[Self::IDX_CAPITAL_GAIN]),
        );
        acc.add(
            &self
                .capital_loss
                .encode(s.capital_loss)
                .bind(&self.keys[Self::IDX_CAPITAL_LOSS]),
        );
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

fn main() {
    hdv!(binary, HDV, 1024);
    let mut rng = MersenneTwister64::default();
    let encoder = AdultEncoder::<HDV>::new(&mut rng);

    println!("Hello World!");
    let dir = "data/ADULT/";

    let Ok(data) = Dataset::load(dir) else {
        println!("Failed to load dataset: {dir}");
        return;
    };
    for (sample, label) in data.train.iter().zip(data.train_labels.iter()) {
        println!("{label:?}, {sample:?}");
        let h = encoder.encode(&sample);
        println!("{}", h.to_braille(80));
    }
}
