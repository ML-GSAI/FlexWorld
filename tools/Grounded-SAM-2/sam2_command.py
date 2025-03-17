import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import grounding_dino.groundingdino.datasets.transforms as T
from PIL import Image
"""
Hyper parameters
"""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Grounded_SAM2():

    def __init__(self):

        SAM2_CHECKPOINT = "./tools/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = "./tools/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = "./tools/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

    def transorm_image(self, image: np.ndarray):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source= Image.fromarray(image).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def __call__(self, img, text, BOX_THRESHOLD = 0.35, TEXT_THRESHOLD = 0.25):
        '''
        img: np.ndarray : np.array(Image.open(img_path).convert("RGB))
        text: str : "computer. table. pad. sofa. cellphone. pillow."
        '''

        sam2_predictor = self.sam2_predictor
        grounding_model = self.grounding_model 

        image_source, image = self.transorm_image(img)

        sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


        # FIXME: figure how does this influence the G-DINO model
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        return detections, class_names, confidences

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        """
        Dump the results in standard format and save as json files
        """

        # def single_mask_to_rle(mask):
        #     rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        #     rle["counts"] = rle["counts"].decode("utf-8")
        #     return rle

        # if DUMP_JSON_RESULTS:
        #     # convert mask into rle format
        #     mask_rles = [single_mask_to_rle(mask) for mask in masks]

        #     input_boxes = input_boxes.tolist()
        #     scores = scores.tolist()
        #     # save the results in standard format
        #     results = {
        #         "image_path": img_path,
        #         "annotations" : [
        #             {
        #                 "class_name": class_name,
        #                 "bbox": box,
        #                 "segmentation": mask_rle,
        #                 "score": score,
        #             }
        #             for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        #         ],
        #         "box_format": "xyxy",
        #         "img_width": w,
        #         "img_height": h,
        #     }
            
        #     with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        #         json.dump(results, f, indent=4)