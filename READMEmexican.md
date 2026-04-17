# Hypervector - Mexican Dollar

This example demonstrates analogy inference in hyperdimensional computing:

> "What is the dollar of Mexico?"

Using hypervector algebra, the system infers that the answer is Mexican peso (mpe).


## Idea

We encode relationships like:

```
    USA → (name, capital, currency)
    Mexico → (name, capital, currency)
```

These are stored as bundled hypervectors:

```
    USA     = name ⊗ usa + capital ⊗ wdc + currency ⊗ usd
    Mexico  = name ⊗ mex + capital ⊗ cdmx + currency ⊗ mpe
```

We then compute a transformation:

```
    T = Mexico ⊗ USA⁻¹
```

and apply it to usd:
```
    T ⊗ usd ≈ mpe
```

Finally, we use cleanup (nearest neighbor search) to recover the closest known symbol.


## What this shows

* Binding (⊗) encodes relationships
* Bundling (+) aggregates facts
* Inverse (⁻¹) enables analogy transfer
* The same computation works across multiple HDV types


## Preliminaries

[Install Rust](https://rust-lang.org/tools/install/).


## Example Run

```bash
cargo run --example mexican

Binary
=========
Nearest HDV is: mpe


Bipolar
=========
Nearest HDV is: mpe


Modular
=========
Nearest HDV is: mpe


ComplexHDV
=========
Nearest HDV is: mpe


RealHDV
=========
Nearest HDV is: mpe
```

## Notes

* All hypervectors are randomly generated but reproducible (fixed seed).
* Different representations (binary, bipolar, complex, etc.) yield the same result.
* Dimensionality affects robustness but not the underlying logic.

## References

1. ["What We Mean When We Say 'What’s the Dollar of Mexico?'" – Pentti Kanerva, 2010](https://aaai.org/papers/02243-2243-what-we-mean-when-we-say-whats-the-dollar-of-mexico-prototypes-and-mapping-in-concept-space/)  
