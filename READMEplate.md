# Hypervector - Inference

Inference example from Tony Plate's [1995 HRR paper](https://www.researchgate.net/publication/5589577_Holographic_Reduced_Representations). Frame HDVs are created from token and role HDVs to represent the meaning
of the sentences:

1. "Mark ate the fish."
1. "Hunger caused Mark to eat the fish."
1. "John ate."
1. "John saw Mark."
1. "John saw the fish."
1. "The fish saw John."

Run
-----

Run example with
```
% cargo run --release --bin main_plate
```

and examine the results with

```
% uv run plot.py
```
#### Binary HDV
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/binary_hdv_combined.png)

#### Bipolar HDV
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/bipolar_hdv_combined.png)

#### Real HDV
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/real_hdv_combined.png)

#### Complex HDV
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/complex_hdv_combined.png)
