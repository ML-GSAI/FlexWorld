import numpy as np
from enum import Enum
import cv2
from PIL import Image
import imageio
import einops

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part

def make_random_irregular_mask(shape, max_angle=4, max_len=200, max_width=100, min_times=4, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=200, max_width=100, min_times=5, max_times=20, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        ''' img: H, W, C
        ret mask: H, W, 1
        '''
        img = img.transpose((2, 0, 1))
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        res =  make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)
        return res.transpose((1, 2, 0)) == 1
    



class VideoMaskGenerator:
    def __init__(self):
        self.img_mask_generator = RandomIrregularMaskGenerator()

    def format_video(self, video):
        '''return video list of H, W, C'''
        if isinstance(video, str):
            video = imageio.mimread(video)
        elif isinstance(video, list):
            assert video[0].shape[2] == 3
        return video
    
    def mask_from_first_frame(self, video):
        '''video: H, W, C, T'''
        video = self.format_video(video)
        mask = self.img_mask_generator(video[0])
        for i in range(len(video)):
            video[i] = (~mask) * video[i]

        return video
    
    def mask_from_all_frame(self, video):
        video = self.format_video(video)
        for i in range(len(video)):
            video[i] = self.img_mask_generator(video[i]) * video[i]
        return video


    def __call__(self, video):
        '''video: T * [H, W, C], or T H W C'''
        video = self.mask_from_all_frame(video)
        return video
        

vid = VideoMaskGenerator()("./cache/real.mp4")
imageio.mimsave("./cache/1_mask.mp4", vid)
# gen = RandomIrregularMaskGenerator()
# img = "./assets/room.png"
# img = np.array(Image.open(img))
# print(img.shape)
# img = img.transpose((2,0,1))
# img_mask = gen(img)
# img_mask = img_mask[0]
# print(img_mask)
# print(img_mask.dtype)
# print(img_mask.shape)
# Image.fromarray((img_mask*255).astype("uint8")).save("./cache/render_0.png")
