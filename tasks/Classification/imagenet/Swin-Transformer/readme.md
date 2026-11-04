# Swin-Transformer-DenoiseRep for Image Classification

This folder contains the implementation of the Swin-Transformer-DenoiseRep for image classification.


## Usage

### Install



- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.7 -y
conda activate swin
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

- Install fused window process for acceleration, activated by passing `--fused_window_process` in the running script
```bash
cd kernels/window_process
python setup.py install #--user
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

### Training

You need to load the trained model before starting to train DenoiseRep.
The trained model from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

For example, we trained `Swinv2-Tiny` with 2 GPU on a single node for 120 epochs:

```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py \
--cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml --data-path <imagenet-path> --batch-size 128 --pretrained swinv2_Tiny_patch4_window8_256.pth
```

### Evaluation

To evaluate:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `Swinv2-Tiny` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml --resume swinv2_tiny_patch4_window8_256_denoise.pth --data-path <imagenet-path>
```
