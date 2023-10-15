import os
import numpy as np
import torch
from PIL import Image
import time

from segment_anything import sam_model_registry, SamPredictor

def initialize_sam_predictor(device_id=0):
    sam_checkpoint_path = os.path.join(os.path.dirname(__file__), "../sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device)
    sam_predictor = SamPredictor(sam_model)
    return sam_predictor

def process_image_and_get_segmentation(predictor, input_image, *bbox_sliders):
    bounding_box = np.array(bbox_sliders)
    image_data = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image_data)

    image_height, image_width, _ = image_data.shape
    input_point = np.array([[image_height // 2, image_width // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bounding_box,
        multimask_output=True
    )

    optimal_mask_index = np.argmax(scores)
    segmentation_mask = masks[optimal_mask_index]
    output_image = np.zeros((image_data.shape[0], image_data.shape[1], 4), dtype=np.uint8)
    output_image[:, :, :3] = image_data
    output_image_bbox = output_image.copy()
    output_image[:, :, 3] = segmentation_mask.astype(np.uint8) * 255
    output_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(output_image_bbox, mode='RGBA')
