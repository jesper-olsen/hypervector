use crate::{Accumulator, HyperVector};
use rand::Rng;
use rand::seq::SliceRandom;

pub struct ScalarEncoder<H: HyperVector> {
    pub min: f32,
    pub max: f32,
    pub basis: Vec<H>,
}

impl<H: HyperVector> ScalarEncoder<H> {
    // calculate `num_levels` correlated basis vectors representing the numbers min..max
    // correlated means 1st vector is more similar to 2nd than to 3rd etc.
    pub fn new<R: Rng>(min: f32, max: f32, num_levels: usize, rng: &mut R) -> Self {
        let v_min = H::random(rng);
        let v_max = H::random(rng);

        // Shuffle indices to pick which bits to swap
        let mut indices: Vec<usize> = (0..H::DIM).collect();
        indices.shuffle(rng);

        let basis = (0..num_levels)
            .map(|i| {
                let frac = i as f32 / (num_levels - 1) as f32;
                let num_to_swap = (frac * H::DIM as f32) as usize;
                let current_indices = &indices[0..num_to_swap];
                v_min.blend(&v_max, current_indices)
            })
            .collect();

        Self { min, max, basis }
    }

    pub fn encode(&self, value: f32) -> &H {
        let normalized = ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
        let index = (normalized * (self.basis.len() - 1) as f32) as usize;
        &self.basis[index]
    }
}

//pub struct TabularEncoder<H: HyperVector> {
//    // Maps column name to its specific ScalarEncoder
//    pub field_encoders: HashMap<String, ScalarEncoder<H>>,
//    // Maps column name to its unique ID vector (the "Key")
//    pub field_keys: HashMap<String, H>,
//}
//
//impl<H: HyperVector> TabularEncoder<H> {
//    /// Initialize encoders based on a schema: (Name, Min, Max, Resolution)
//    pub fn new(
//        schema: Vec<(String, f32, f32, usize)>,
//        rng: &mut impl RngCore
//    ) -> Self {
//        let mut field_encoders = HashMap::new();
//        let mut field_keys = HashMap::new();
//
//        for (name, min, max, levels) in schema {
//            field_encoders.insert(
//                name.clone(),
//                ScalarEncoder::new(min, max, levels, rng)
//            );
//            // Each column gets a completely random orthogonal key
//            field_keys.insert(name, H::random(rng));
//        }
//
//        Self { field_encoders, field_keys }
//    }
//
//    /// Encodes a single row/record into one HyperVector
//    pub fn encode_record(&self, record: &HashMap<String, f32>) -> H {
//        let mut acc = H::Accumulator::default();
//
//        for (name, value) in record {
//            if let (Some(enc), Some(key)) = (self.field_encoders.get(name), self.field_keys.get(name)) {
//                // Scalar value -> Level Vector
//                let val_v = enc.encode(*value);
//                // Bind Level Vector with Column Key
//                let bound = val_v.bind(key);
//                acc.add(&bound, 1.0);
//            }
//        }
//
//        acc.finalize()
//    }
//}

pub struct TabularEncoder<H: HyperVector> {
    // Vectors indexed by the column position in the CSV
    pub field_encoders: Vec<ScalarEncoder<H>>,
    pub field_keys: Vec<H>,
}

impl<H: HyperVector> TabularEncoder<H> {
    /// Initialize from a slice of (min, max, resolution)
    pub fn new(schema: &[(f32, f32, usize)], rng: &mut impl Rng) -> Self {
        let mut field_encoders = Vec::with_capacity(schema.len());
        let mut field_keys = Vec::with_capacity(schema.len());

        for &(min, max, levels) in schema {
            field_encoders.push(ScalarEncoder::new(min, max, levels, rng));
            field_keys.push(H::random(rng));
        }

        Self {
            field_encoders,
            field_keys,
        }
    }

    /// Encodes a row of features. Assumes row.len() == schema.len()
    pub fn encode_row(&self, row: &[f32]) -> H {
        let mut acc = H::Accumulator::new();

        for (i, &value) in row.iter().enumerate() {
            // Direct index access - no hashing or string matching
            let val_v = self.field_encoders[i].encode(value);
            let bound = val_v.bind(&self.field_keys[i]);
            acc.add(&bound, 1.0);
        }

        acc.finalize()
    }
}

#[cfg(test)]
mod encoding_tests {
    use super::*;
    use crate::binary_hdv::BinaryHDV;
    use mersenne_twister_rs::MersenneTwister64;

    #[test]
    fn test_scalar_correlation_gradient() {
        let mut mt = MersenneTwister64::new(42);
        let num_levels = 10;

        let encoder = ScalarEncoder::<BinaryHDV<16>>::new(0.0, 10.0, num_levels, &mut mt);

        // Get vectors for 0, 1, and 10
        let v0 = encoder.encode(0.0);
        let v1 = encoder.encode(1.1); // Should map to index 1
        let v9 = encoder.encode(10.0); // Should map to index 9

        let dist_0_1 = v0.distance(v1);
        let dist_0_9 = v0.distance(v9);

        println!("Dist(0, 1): {dist_0_1}");
        println!("Dist(0, 9): {dist_0_9}");

        // Verify: Adjacent values are closer than distant values
        assert!(dist_0_1 < dist_0_9);
        // Verify: v0 and v9 should be nearly orthogonal (~0.5)
        assert!(dist_0_9 > 0.4);
    }

    #[test]
    fn test_clamping() {
        let mut mt = MersenneTwister64::new(42);
        let encoder = ScalarEncoder::<BinaryHDV<16>>::new(0.0, 10.0, 5, &mut mt);

        // Values outside range should not panic and should clamp to boundaries
        let low = encoder.encode(-5.0);
        let high = encoder.encode(15.0);

        assert_eq!(low, &encoder.basis[0]);
        assert_eq!(high, &encoder.basis[4]);
    }
}
