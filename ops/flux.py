import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/FLUX-Controlnet-Inpainting'
sys.path.insert(0,reference)

from flux_command import FLUX