import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .fvd_tool.fvd import preprocess, frechet_distance
from .fvd_tool.pytorch_i3d import InceptionI3d
from .fvd_tool.dataset import load_videos_given_path

TARGET_RESOLUTION = (224, 224)

def load_fvd_model(device, pretrained_path=None):
        i3d = InceptionI3d(400, in_channels=3).to(device)
        if pretrained_path:
            i3d_path = pretrained_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            i3d_path = os.path.join(current_dir, 'fvd', 'i3d_pretrained_400.pt')
        i3d.load_state_dict(torch.load(i3d_path, map_location=device))
        i3d.eval()
        return i3d

class ScorerFVD():

    def __init__(self, device=None, pretrained_path=None):
        self.device = device
        self.i3d = load_fvd_model(device, pretrained_path)    
        
    def _preprocessing(self, frames_batch):
        """
        preprocess frame for FVD calculation.
        input:
            frames_batch: [0,255], [b, t, h, w, c] 
        
        output:
            [-1, 1], [b, t, h, w, c] 
        """
        # frames_batch = (frames_batch / 127.5 - 1.0)
        # return frames_batch 
        
        # TODO: add crop and resize ??
        return preprocess(frames_batch, TARGET_RESOLUTION)
        
    
            
    @torch.no_grad()
    def get_fvd_logits(self, frames_batch, batch_size=16, show_progress=False):
        """
        input:
            frames_batch: [0,255], [b, t, h, w, c] 
        
        output:
            fvd_logits: [b, 400]
        """
        frames_batch = self._preprocessing(frames_batch)
        
        fvd_logits = [] 
        for i in tqdm(range(0, len(frames_batch), batch_size), disable=not show_progress):
            batch_input = frames_batch[i:i+batch_size].to(self.device)
            batch_logits = self.i3d(batch_input).cpu().detach()
            fvd_logits.append(batch_logits)
            
        fvd_logits = torch.cat(fvd_logits, dim=0)
        return fvd_logits
    
    
    def fvd(self, fake_batch, real_batch, batch_size=16, show_progress=True):
        """
        input:
            fake_batch: [0,255], [b, t, h, w, c] 
            real_batch: [0,255], [b, t, h, w, c] 
        
        output:
            fvd_score
        """
        fake_logits = self.get_fvd_logits(fake_batch, batch_size=batch_size, show_progress=show_progress)
        real_logits = self.get_fvd_logits(real_batch, batch_size=batch_size, show_progress=show_progress)
        
        fvd_score = frechet_distance(fake_logits.to(self.device), real_logits.to(self.device))
        return fvd_score.cpu().detach()
    
        
        