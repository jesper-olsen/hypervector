# Hypervector - Inference

Inference example from [1]. Frame HDVs are created from token and role HDVs to represent the meaning
of the sentences:

1. "Mark ate the fish."
1. "Hunger caused Mark to eat the fish."
1. "John ate."
1. "John saw Mark."
1. "John saw the fish."
1. "The fish saw John."

1. ["Holographic Reduced Representations", Tony Plate, IEEE Transactions on Neural Networks, February, 1995, 6(3):623-41](https://www.researchgate.net/publication/5589577_Holographic_Reduced_Representations)

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
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/BinaryHDV_objects.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/BinaryHDV_sentences.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/BipolarHDV_objects.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/BipolarHDV_sentences.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/RealHDV_objects.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/RealHDV_sentences.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/ComplexHDV_objects.png)
![PNG](https://github.com/jesper-olsen/hypervector/blob/master/ASSETS/ComplexHDV_sentences.png)
