## INSTALLATION

### Environment Setup Instructions
Basic environment
```bash
conda create -n FlexWorld python=3.11
conda activate FlexWorld
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2
# other dependencies
pip install -r requirements.txt

# for video generate
cd ./tools/CogVideo
pip install -r requirements.txt
cd ../..

# for super-resolution
pip install basicsr
# Note: When using super-resolution, you need to modify the code in package `basicsr.data`. Change `torchvision.transforms.functional_tensor` to `torchvision.transforms.functional` in `degradation.py`.
```

### Pretrained model
```bash
# DUSt3R and MASt3R
mkdir ./tools/dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ./tools/dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./tools/dust3r/checkpoints

# CogVideoX-SAT
pip install -U huggingface_hub
huggingface-cli download GSAI-ML/FlexWorld --local-dir ./tools/CogVideo/checkpoints

# Real-ESRGAN
mkdir ./tools/Real_ESRGAN/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./tools/Real_ESRGAN/weights
```