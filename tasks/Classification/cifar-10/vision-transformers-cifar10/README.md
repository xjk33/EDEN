# ViT-cifar10-DenoiseRep

We apply our propsoed [DenoiseRep](../../../denoiserep_op/) to [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10), and get stable improvements.

# Installation
```bash
(we use /torch 1.8.0 /torchvision 0.9.0 /timm 0.6.12 /cuda 10.1 / 24G or 48G RTX3090 for training and evaluation.)
cd denoiserep_op
bash make.sh
pip show denoiserep
pip install -r requirements.txt
```


# Load the trained model
Firstly, you need to train the baseline model
```bash
python train_cifar10.py
```
Next, you need to load the trained model before starting to train DenoiseRep.
```python
# train_cifar10_denoise.py:L159
checkpoint = torch.load('/pretrain_model_path')
net.load_state_dict(checkpoint['model'])
```


# Train
```bash
python train_cifar10_denoise.py
```


# Results
| Model              | Accuracy (%) |
|--------------------|--------------|
| ViT patch=4 (Baseline)     | 85.57         |
| ViT patch=4 (+_DenoiseRep_)   | 86.21 [(model)](https://drive.google.com/file/d/1exsexxqnoG7hwifh4GkFtO6XO_HEq3U8/view?usp=sharing)      |