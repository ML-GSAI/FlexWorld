import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/Grounded-SAM-2'
sys.path.insert(0,reference)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
from PIL import Image
import imageio

from sam2_command import Grounded_SAM2


def crop_to_square(image_np, output_path='./cache/1_rgba.png'):
    """
        Crop the image to a square shape around the center of the non-zero pixels.
    """
    non_zero_coords = np.argwhere(image_np[..., 0] != 0)
    if non_zero_coords.size == 0:
        raise ValueError("No non-zero pixels found in the image!")

    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0)

    center = (top_left + bottom_right) // 2
    height, width = bottom_right - top_left + 1
    side_length = max(height, width)

    half_side = side_length // 2
    top = center[0] - half_side
    left = center[1] - half_side
    bottom = center[0] + half_side
    right = center[1] + half_side

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - image_np.shape[0])
    pad_right = max(0, right - image_np.shape[1])

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, image_np.shape[0])
    right = min(right, image_np.shape[1])

    cropped_image_np = image_np[top:bottom, left:right]

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        cropped_image_np = np.pad(
            cropped_image_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )

    cropped_image = Image.fromarray(cropped_image_np.astype(np.uint8))

    alpha = np.ones((cropped_image_np.shape[0], cropped_image_np.shape[1]), dtype=np.uint8) * 255
    alpha[np.all(cropped_image_np == 0, axis=-1)] = 0
    cropped_image.putalpha(Image.fromarray(alpha))

    cropped_image.save(output_path)
    return np.array(cropped_image)
    
class Obj_Inform():
    def __init__(self, obj, mask, label, confidence, ent_type='fg',idx=None):
        self.obj = obj
        self.mask = mask
        self.label = label
        self.confidence = confidence
        self.idx=idx
        self.ent_type=ent_type
        self.cropped = None
        if ent_type =='fg':
            self.cropped = crop_to_square(obj)
        
class VideoMaskGenerator():
    def __init__(self):
        sam2_checkpoint = "./tools/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = sam2_model
        self.prompt = []


    def load_video(self, video_path):
        self.video_path = video_path
        self.video = imageio.mimread(video_path)
        self.state = self.predictor.init_state(video_path=video_path)
    
    def seg_first_frame(self, newpts, frameidx=0):
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frameidx,
            obj_id=1,
            points=newpts,
            labels=labels,
        )
        return out_mask_logits
    

    def auto_select(self):
        pass

    def propogate(self):
        pass

    def compute_image_score(self, mask):
        pass

    def compute_video_score(self, video_mask):
        pass


class Segment_Tool():
    def __init__(self):

        sam2_checkpoint = "./tools/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)

    def auto_point_select(self,input_image,iters=20):
        h, w =input_image.shape[:2]
        mask = np.zeros((h,w))
        x_coords, y_coords = np.meshgrid(np.linspace(0.1 * h,0.9* h, 10), np.linspace(0.1 * w,0.9 * w, 10))
        all_points = np.column_stack((x_coords.ravel(), y_coords.ravel())).astype(np.uint32)
        
        for i in range(iters):
            points = all_points[mask[all_points[:, 0], all_points[:, 1]] == 0]
            if points.shape[0] == 0:
                # No more points to select
                break
            # print(points.shape[0])
            idx = int(points.shape[0]/2)
            point = (points[idx][1],points[idx][0])
            print(point)
            
            mask_img , new_mask , _ = self(input_image, input_point=point)
            
            mask = np.maximum(mask, new_mask)
            mask[points[idx][0],points[idx][1]] = 1.0
            
            Image.fromarray((mask_img).astype(np.uint8)).save(f'cache/mask_{i}.png')
        # result_image = Image.fromarray((mask * 255).astype(np.uint8))
        # result_image.save('cache/mymask.png')
            
    def __call__(self, input_image, input_point=(500, 375)):
        """
            input image is a np.ndarray of shape (H, W, 3) 
            input_point: W, H
        """
        # image = Image.open('./cache/test.jpg')
        # image = np.array(image.convert("RGB"))

        predictor = self.predictor
        predictor.set_image(input_image)

        input_point = np.array([input_point])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        img_raw = input_image
        img_mask = masks[0]
        masked_img = img_raw * img_mask[..., None]
        cropped = crop_to_square(masked_img)
        return masked_img, img_mask , cropped 


class Grounded_SAM2_Tool():
    def __init__(self, device='cuda'):
        self.grounded_sam2 = Grounded_SAM2()
        
    def __call__(self, img, text_fg, text_bg, BOX_THRESHOLD = 0.35, TEXT_THRESHOLD = 0.25):
        """
        img: np.ndarray of shape (H, W, 3) 
        """
        text = text_fg + ' ' + text_bg
        fg = text_fg.replace('.', '').split()
        # bg = text_bg.replace('.', '').split()
        detections, labels, confidences = self.grounded_sam2(img, text, BOX_THRESHOLD = BOX_THRESHOLD, TEXT_THRESHOLD = TEXT_THRESHOLD)
        
        img_raw = img
        masks=detections.mask
        masks=masks.astype(np.float32)
        fg_masks, bg_masks = [], []
        fgs, bgs = [], []
        
        for i, label in enumerate(labels):
            img_mask = masks[i]
            masked_img = img_raw * img_mask[..., None]
            if label in fg:
                fg_masks.append(img_mask)
                fgs.append(Obj_Inform(masked_img, img_mask, label, confidences[i],ent_type='fg',idx=i))
            else:
                # bg_masks.append(img_mask)
                bgs.append(Obj_Inform(masked_img, img_mask, label, confidences[i],ent_type='bg',idx=i))
                
        fg_masks = np.stack(fg_masks,axis=0)
        bg_mask = np.logical_not(np.any(fg_masks,axis=0)).astype(np.float32)
        
        Image.fromarray((bg_mask*255).astype(np.uint8)).save('cache/bg_mask.png')
        # background = img_raw * bg_mask[..., None]
        
        return fgs, bgs, bg_mask