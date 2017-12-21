# GAN CelebA

# Image Generate

``` sh
python add_attr.py
```

# Model Training

``` sh
python download.py
python train_gen.py
python train_enc.py --gen <trained generator model>
python get_vectors.py --enc <trained encoder model>
```
