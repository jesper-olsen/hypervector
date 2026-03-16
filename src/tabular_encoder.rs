use crate::data::{ColumnType, DatasetMetadata, Sample, SampleValue};
use crate::encoding::ScalarEncoder;
use crate::{Accumulator, HyperVector};
use rand::Rng;
use std::collections::HashMap;

pub enum FieldEncoder<H: HyperVector> {
    Scalar(ScalarEncoder<H>),
    Categorical(Vec<H>), // Index corresponds to the Vocabulary ID
    None,                // For skipped columns or missing data
}

pub struct TabularEncoder<H: HyperVector> {
    pub field_encoders: Vec<FieldEncoder<H>>,
    pub field_keys: Vec<H>,
}

impl<H: HyperVector> TabularEncoder<H> {
    pub fn from_metadata(
        metadata: &DatasetMetadata,
        // (min, max, resolution) for numeric columns
        numeric_params: &HashMap<usize, (f32, f32, usize)>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut field_encoders = Vec::new();
        let mut field_keys = Vec::new();

        for (idx, col_type) in metadata.column_types.iter().enumerate() {
            // 1. Generate the static Key for this column
            field_keys.push(H::random(rng));

            // 2. Generate the Value Encoder
            match col_type {
                ColumnType::Numeric => {
                    let (min, max, res) =
                        numeric_params.get(&idx).cloned().unwrap_or((0.0, 1.0, 100)); // Defaults
                    field_encoders
                        .push(FieldEncoder::Scalar(ScalarEncoder::new(min, max, res, rng)));
                }
                ColumnType::Categorical => {
                    // For categories, we generate N random vectors
                    // where N is the total size of your interning vocabulary.
                    let num_categories = metadata.vocabulary.len();
                    let cat_vectors = (0..num_categories).map(|_| H::random(rng)).collect();
                    field_encoders.push(FieldEncoder::Categorical(cat_vectors));
                }
                ColumnType::Mixed => {
                    // Handle as Categorical for safety, or implement custom logic
                    field_encoders.push(FieldEncoder::None);
                }
            }
        }

        Self {
            field_encoders,
            field_keys,
        }
    }

    pub fn encode(&self, sample: &Sample, exclude: &[usize]) -> H {
        let mut acc = H::Accumulator::default();
        for (i, value) in sample.iter().enumerate() {
            if exclude.contains(&i) {
                continue;
            }
            let val_hv = match (value, &self.field_encoders[i]) {
                (SampleValue::Numeric(f), FieldEncoder::Scalar(enc)) => Some(enc.encode(*f as f32)),
                (SampleValue::String(id), FieldEncoder::Categorical(vec)) => vec.get(*id),
                _ => None,
            };
            if let Some(v) = val_hv {
                acc.add(&v.bind(&self.field_keys[i]), 1.0);
            }
        }
        acc.finalize()
    }
}
