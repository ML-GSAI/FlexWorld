import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class RealESRGAN():
    def __init__(self,device='cuda'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_path= './tools/Real_ESRGAN/weights/RealESRGAN_x4plus.pth'
        dni_weight = None
        
        self.upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device)

    def __call__(self, input_image, outscale=4.0):
        """
            input_image: np.ndarray
        """
        image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        output, _ = self.upsampler.enhance(image, outscale=outscale)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
    