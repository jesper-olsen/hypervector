use hypervector::binary_hdv::BinaryHDV;
use hypervector::data::{LoadedSplitDataset, Sample, SampleValue, load_train_test_csv};
use hypervector::encoding::TabularEncoder;
use hypervector::{Accumulator, HyperVector};
use mersenne_twister_rs::MersenneTwister64;

type HDV = BinaryHDV<1024>; // 1024-bit vectors

fn encode_dataset(encoder: &TabularEncoder<HDV>, data: &[Sample]) -> (Vec<HDV>, Vec<usize>) {
    let mut hvs: Vec<HDV> = Vec::with_capacity(data.len());
    let mut labels: Vec<usize> = Vec::with_capacity(data.len());
    for sample in data {
        let n = sample.len() - 1;
        let row: Vec<f32> = sample[..n]
            .iter()
            .map(|v| {
                if let SampleValue::Numeric(a) = *v {
                    a as f32
                } else {
                    panic!("expected numeric")
                }
            })
            .collect();

        // label is the quality score string, e.g. "5", "6", "7"
        let label_idx = if let SampleValue::String(idx) = sample[n] {
            idx
        } else {
            panic!("expected categorical label")
        };

        let encoded = encoder.encode_row(&row);
        hvs.push(encoded);
        labels.push(label_idx);
    }
    (hvs, labels)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fname_train = "DATA/WINEQUALITY/winequality_red_train.csv";
    let fname_test = "DATA/WINEQUALITY/winequality_red_test.csv";

    let dataset: LoadedSplitDataset = load_train_test_csv(fname_train, fname_test, None, true)?;

    let schema: &[(f32, f32, usize)] = &[
        (4.0, 16.0, 64),
        (0.1, 1.6, 64),
        (0.0, 1.0, 64),
        (1.0, 16.0, 64),
        (0.01, 0.62, 64),
        (1.0, 72.0, 64),
        (6.0, 290.0, 64),
        (0.99, 1.004, 64),
        (2.7, 4.1, 64),
        (0.3, 2.0, 64),
        (8.0, 15.0, 64),
    ];

    let mut rng = MersenneTwister64::new(42);
    let encoder = TabularEncoder::<HDV>::new(schema, &mut rng);

    let num_classes = 2usize; // indices 0..=8 map to quality scores

    let (train_hvs, train_labels) = encode_dataset(&encoder, &dataset.train_data);

    type Acc = <HDV as HyperVector>::Accumulator;
    let mut class_accumulators: Vec<Acc> = (0..num_classes).map(|_| Acc::new()).collect();

    train_hvs
        .iter()
        .zip(train_labels.iter())
        .for_each(|(h, label)| class_accumulators[*label].add(h, 1.0));

    // Finalize prototype ("class hypervector") for each class
    let class_vectors: Vec<HDV> = class_accumulators
        .into_iter()
        .map(|mut acc| acc.finalize())
        .collect();

    // Evaluate on test set
    let mut correct = 0usize;
    let mut total = 0usize;

    let (test_hvs, test_labels) = encode_dataset(&encoder, &dataset.test_data);
    for (encoded, true_label) in test_hvs.iter().zip(test_labels.iter()) {
        // Nearest-prototype classification
        let pred = class_vectors
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance(encoded)
                    .partial_cmp(&b.distance(encoded))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        if pred == *true_label {
            correct += 1;
        }
        total += 1;
    }

    println!("Accuracy: {:.1}%", 100.0 * correct as f32 / total as f32);
    Ok(())
}
