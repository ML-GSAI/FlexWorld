import os
from decord import VideoReader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from PIL import Image
from einops import rearrange
class FVDTestDataset(Dataset):
    def __init__(self, data_path, sample_size=256):
        filenames = sorted(os.listdir(data_path))
        self.length = len(filenames)
        self.data_path = data_path
        self.filenames = filenames

        if isinstance(sample_size, int):
            sample_size = tuple([int(sample_size)] * 2)
        else:
            raise EOFError

        self.pixel_transforms = transforms.Compose([
            transforms.CenterCrop(sample_size),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_dir = os.path.join(self.data_path, self.filenames[idx])
        video_reader = VideoReader(video_dir)

        sample_stride = 1
        video_length = len(video_reader)
        clip_length = min(video_length, (video_length - 1) * sample_stride + 1)
        start_idx = 0
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, video_length, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = self.pixel_transforms(pixel_values)  # [T, C, H, W]
        return pixel_values
    
    
def load_videos_given_path(path, resolution = 256):
    dataset = FVDTestDataset(data_path=path, sample_size=resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=32,
        drop_last=False,
    )
    videos = []
    for step, pixel_values in enumerate(dataloader):
        videos.append(pixel_values)
    videos = torch.cat(videos)
    videos = rearrange(videos, 'b t c h w -> b t h w c')
    return videos.numpy()