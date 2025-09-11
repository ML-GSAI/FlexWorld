# FlexWorld
[![arXiv](https://img.shields.io/badge/arXiv-2503.13265-red.svg)](https://arxiv.org/abs/2503.13265)
[![deploy](https://img.shields.io/badge/🤗%20Hugging%20Face%20-FlexWorld-FFEB3B)](https://huggingface.co/GSAI-ML/FlexWorld)
[![deploy](https://img.shields.io/badge/Project%20Page-black)](https://ml-gsai.github.io/FlexWorld/)

This is the official PyTorch implementation of *[FlexWorld: Progressively Expanding 3D Scenes for Flexiable-View Synthesis](http://arxiv.org/abs/2503.13265)*.

## Update 
- [2025-5-21]: Add training code and data preperation. 

## Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## Usage
Static scene video generation given an image and a camera trajectory:

```bash
python video_generate.py --input_image_path ./assets/room.png --output_dir ./results-single-traj
```

You can pass in `traj` argument to specify camera movements, the basic movements is defined in "ops/utils/all_traj.py". The supported camera movements includes `["up","down","left","right","forward","backward","rotate_left","rotate_right"]`.

```bash
python video_generate.py --input_image_path ./assets/room.png --output_dir ./results-single-traj --traj backward
```

You can also generate videos share the same camera trajectories with those in DL3DV and Re10K. Just pass the video path to `traj` arguments.
```bash
python video_generate.py --input_image_path ./assets/room.png --output_dir ./results-single-traj --traj ./path_to_dl3dv/1.mp4
```



A flexible-view 360° scene generation given an image.
```bash
# You are free to modify the corresponding YAML configuration file by name in `./configs/examples`.
python main_3dgs.py --name room2
```

## Visualization

First running:
```bash
python 3dgs_viewer.py
```
then visit `127.0.0.1:8000` to freely explore the generated scene in the current directory. The script will scan the `ply` file recursively, please doing this after the generation.

## Dataset Preperation

1. Download dataset to local dir following [DL3DV repo](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P). You may download only part of them, like 1K.

2. Prepare 3DGS from DL3DV dataset, you can first download colmap annotation from [DL3DV colmap annotation](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-ColmapCache) and then do reconstruction following [Gaussian Splatting repo](https://github.com/graphdeco-inria/gaussian-splatting). The final output will listed like:
```
- output/
  - 001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f
    - point_cloud
        - iteration_7000
            - point_cloud.ply
  - 0003dc82473fd52c53dcbdc2deb4e6e9c3548d6f8c9b03f9ea8d3c7d3ce33546
    - point_cloud
        - iteration_7000
            - point_cloud.ply
```

3. Run following to generate broken video constructed by 3DGS.
```bash
# The path here is an example.
python gen_dataset.py --dataset_path ./DL3DV/DL3DV-10K/1K --output_path ./DL3DV/processed --gs_path ./gaussian-splatting/output 
```

4. Run following to label the video constructed.
```bash
# The path here is an example.
python label_dataset.py --input_path ./DL3DV/processed --output_path ./train_data_v2v
```

## Training 
1. Change following lines in "./tools/CogVideo/configs/sft_v2v.yaml".
```yaml
args:
  checkpoint_activations: True 
  experiment_name: lora-disney # your save folder name 
  mode: finetune
  load: "xxx/CogVideoX-5B-I2V-SAT/transformer" # path to transformer original checkpoints
  save: "./ckpts_5b" # path to save dir.
  train_data: [ "train_data_v2v" ] # Train data path
  valid_data: [ "train_data_v2v" ] # Validation data path, can be the same as train_data(no recommended)
```

2. Run training script
```bash
cd ./tools/CogVideo/
bash train_video_v2v.py
```

## ToDo List
- [x] A user manual for our camera trajectory, offering support for more flexible trajectory inputs and accommodating a wider variety of trajectory types (such as RealEstate camera input and DL3DV-10K camera input).
- [x] A 3DGS viewer for generated results.
- [x] Training code for video diffusion model.

## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [CogVideoX](https://github.com/THUDM/CogVideo)
- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R](https://github.com/naver/mast3r)
- [ViewCrafter](https://github.com/Drexubery/ViewCrafter)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [VistaDream](https://github.com/WHU-USI3DV/VistaDream)
- [LucidDreamer](https://github.com/luciddreamer-cvlab)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

## Citation

```
@misc{chen2025flexworldprogressivelyexpanding3d,
      title={FlexWorld: Progressively Expanding 3D Scenes for Flexiable-View Synthesis}, 
      author={Luxi Chen and Zihan Zhou and Min Zhao and Yikai Wang and Ge Zhang and Wenhao Huang and Hao Sun and Ji-Rong Wen and Chongxuan Li},
      year={2025},
      eprint={2503.13265},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.13265}, 
}
```
