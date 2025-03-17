import cv2
import numpy as np
import os
import sys
from .pano_tool import Equirec2Perspec as E2P
from .pano_tool import multi_Perspec2Equirec as m_P2E
import uuid
from PIL import Image
from tqdm import tqdm


def video2pano(image_paths,out_dir,fov=90):
    
    pers = [cv2.imread(image_path) for image_path in image_paths]
    
    ee = m_P2E.Perspective(pers,
                            [[fov, 0, 0], [fov, 45, 0], [fov, 90, 0], [fov, 135, 0],
                             [fov, 180, 0], [fov, 225, 0], [fov, 270, 0], [fov, 315, 0]]
                            )

    new_pano = ee.GetEquirec(2048, 4096)
    resized_pano = cv2.resize(new_pano[650:-650], (1280, 576), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), resized_pano.astype(np.uint8))
    return new_pano