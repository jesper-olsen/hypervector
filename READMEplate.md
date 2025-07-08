# hypervector

Inference example from [1]. Frame HDVs are created from token and role HDVs to represent the meaning
of the sentences:

+       "Mark ate the fish."
+       "Hunger caused Mark to eat the fish."
+       "John ate."
+       "John saw Mark."
+       "John saw the fish."
+       "The fish saw John."

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

![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/BinaryHDV_objects.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/BinaryHDV_sentences.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/BipolarHDV_objects.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/BipolarHDV_sentences.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/RealHDV_objects.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/RealHDV_sentences.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/ComplexHDV_objects.png)
![PNG](https://raw.githubusercontent.com/jesper-olsen/hypervector/main/ASSETS/ComplexHDV_sentences.png)
