// Implementes the example from
// "Holographic Reduced Representations", Tony Plate,
// IEEE Transactions on Neural Networks, February, 1995, 6(3):623-41
// https://www.researchgate.net/publication/5589577_Holographic_Reduced_Representations

use hypervector::HyperVector;
use hypervector::binary_hdv::BinaryHDV;
use hypervector::bipolar_hdv::BipolarHDV;
use hypervector::complex_hdv::ComplexHDV;
use hypervector::real_hdv::RealHDV;
use mersenne_twister_rs::MersenneTwister64;
use std::any::type_name;

use std::fs::File;
use std::io::Write;

fn write_confusion_csv<T: HyperVector>(filename: &str, items: &[(T, &str)]) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    // Write header
    write!(file, ",")?;
    for (_, lab) in items {
        write!(file, "{},", lab.trim())?;
    }
    writeln!(file)?;

    // Write matrix
    for (t1, l1) in items {
        write!(file, "{},", l1.trim())?;
        for (t2, _) in items {
            write!(file, "{:.4},", t1.distance(t2))?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn print_confusions<T: HyperVector>(items: &[(T, &str)]) {
    // Print header row
    print!("{:10}", ""); // empty top-left corner
    for (_, lab) in items {
        print!("{:>8}", lab);
    }
    println!();

    // Print matrix
    for (t1, l1) in items {
        print!("{:>8}", l1); // row label
        for (t2, _) in items {
            print!("{:8.2}", t1.distance(t2)); // align all numbers right with 3 decimals
        }
        println!();
    }
    println!();
}

pub fn plate<T: HyperVector>(fname_prefix: &str) -> std::io::Result<()> {
    println!("\nHDV Type is: {}", type_name::<T>());

    let mut mt = MersenneTwister64::new(42);

    let agent = T::random(&mut mt);
    let agent_cause = T::random(&mut mt);
    let agent_eat = T::random(&mut mt);
    let agent_see = T::random(&mut mt);
    let being = T::random(&mut mt);
    let bread = T::random(&mut mt);
    let cause = T::random(&mut mt);
    let eat = T::random(&mut mt);
    let fish = T::random(&mut mt);
    let food = T::random(&mut mt);
    let id_bread = T::random(&mut mt);
    let id_fish = T::random(&mut mt);
    let id_hunger = T::random(&mut mt);
    let id_john = T::random(&mut mt);
    let id_luke = T::random(&mut mt);
    let id_mark = T::random(&mut mt);
    let id_paul = T::random(&mut mt);
    let id_thirst = T::random(&mut mt);
    let object = T::random(&mut mt);
    let object_cause = T::random(&mut mt);
    let object_eat = T::random(&mut mt);
    let object_see = T::random(&mut mt);
    let person = T::random(&mut mt);
    let see = T::random(&mut mt);
    let state = T::random(&mut mt);
    let id_eat_agent = T::random(&mut mt);
    let id_object_cause = T::random(&mut mt);
    let id_eat_object = T::random(&mut mt);
    let id_see_object = T::random(&mut mt);
    let id_agent_cause = T::random(&mut mt);
    let id_see_agent = T::random(&mut mt);

    let mark = T::acc(&[&being, &person, &id_mark]);
    let john = T::acc(&[&being, &person, &id_john]);
    let paul = T::acc(&[&being, &person, &id_paul]);
    let luke = T::acc(&[&being, &person, &id_luke]);
    let the_fish = T::acc(&[&food, &fish, &id_fish]);
    let the_bread = T::acc(&[&food, &bread, &id_bread]);
    let hunger = T::acc(&[&state, &id_hunger]);
    let thirst = T::acc(&[&state, &id_thirst]);

    let agent_eat = T::acc(&[&agent, &id_eat_agent]);
    let obj_eat = T::acc(&[&object, &id_eat_object]);

    let sentences = [
        "Mark ate the fish.",
        "Hunger caused Mark to eat the fish.",
        "John ate.",
        "John saw Mark.",
        "John saw the fish.",
        "The fish saw John.",
    ];

    let s1 = T::acc(&[&eat, &agent_eat.bind(&mark), &object_eat.bind(&the_fish)]);
    let s2 = T::acc(&[&cause, &agent_cause.bind(&hunger), &object_cause.bind(&s1)]);
    let s3 = T::acc(&[&eat, &agent_eat.bind(&john)]);
    let s4 = T::acc(&[&see, &agent_see.bind(&john), &object_see.bind(&mark)]);
    let s5 = T::acc(&[&see, &agent_see.bind(&john), &object_see.bind(&the_fish)]);
    let s6 = T::acc(&[&see, &agent_see.bind(&the_fish), &object_see.bind(&john)]);

    let l = [
        (mark, "mark   "),
        (john, "john   "),
        (paul, "paul   "),
        (luke, "luke   "),
        (fish, "fish   "),
        (bread, "bread  "),
        (hunger, "hunger "),
        (thirst, "thirst "),
    ];

    print_confusions(&l);
    if fname_prefix.len() > 0 {
        let filename = format!("{fname_prefix}_objects.csv");
        write_confusion_csv(&filename, &l)?;
    }

    for (i, s) in sentences.iter().enumerate() {
        println!("s{}: {s}", i + 1);
    }

    let l = [
        (s1, "s1     "),
        (s2, "s2     "),
        (s3, "s3     "),
        (s4, "s4     "),
        (s5, "s5     "),
        (s6, "s6     "),
    ];
    print_confusions(&l);
    if fname_prefix.len() > 0 {
        let filename = format!("{fname_prefix}_sentences.csv");
        write_confusion_csv(&filename, &l)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    plate::<BipolarHDV<1024>>("RESULTS/hdv_bipolar")?;
    plate::<BinaryHDV<16>>("RESULTS/hdv_binary")?; // 16*64 = 1024
    plate::<RealHDV<512>>("RESULTS/hdv_real")?;
    plate::<ComplexHDV<512>>("RESULTS/hdv_complex")?;
    Ok(())
}
