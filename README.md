# FlexWorld
[![arXiv](https://img.shields.io/badge/arXiv-2503.13265-red.svg)](https://arxiv.org/abs/2503.13265)
[![deploy](https://img.shields.io/badge/🤗%20Hugging%20Face%20-FlexWorld-FFEB3B)](https://huggingface.co/GSAI-ML/FlexWorld)
[![deploy](https://img.shields.io/badge/Project%20Page-black)](https://ml-gsai.github.io/FlexWorld/)

This is the official PyTorch implementation of *[FlexWorld: Progressively Expanding 3D Scenes for Flexiable-View Synthesis](http://arxiv.org/abs/2503.13265)*.


## Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## Usage
Static scene video generation given an image and a camera trajectory:

```bash
# You can utilize our CamPlanner class to freely construct the desired trajectory at line 13 in the `video_generate.py` file.
python video_generate.py --input_image_path ./assets/room.png --output_dir ./results-single-traj
```

A flexible-view 360° scene generation given an image.
```bash
# You are free to modify the corresponding YAML configuration file by name in `./configs/examples`.
python main_3dgs.py --name room2
```

## ToDo List
- [ ] A user manual for our camera trajectory, offering support for more flexible trajectory inputs and accommodating a wider variety of trajectory types (such as RealEstate camera input and DL3DV-10K camera input).
- [ ] A 3DGS viewer for generated results.
- [ ] Training code for video diffusion model.

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