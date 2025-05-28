import sys
from typing import Optional

import numpy as np
import torch
import cv2

from src.project_root import PROJECT_ROOT
SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "sam"  # Get the absolute path to `submodules/sam`
if SAM_MODULE_PATH not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from submodules.sam.segment_anything import SamPredictor, sam_model_registry


def create_sam_predictor(device: torch.device, model_type: str = "vit_b"):
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "sam"
    if model_type == "vit_b":
        checkpoint = checkpoint_path / "sam_vit_b_01ec64.pth"
    elif model_type == "vit_l":
        checkpoint = checkpoint_path / "sam_vit_l_0b3195.pth"
    elif model_type == "vit_h":
        checkpoint = checkpoint_path / "sam_vit_h_4b8939.pth"
    else:
        raise ValueError(f"invalid model_type {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def extract_and_process_slice_like_sam(input_image: np.ndarray, slice_idx: int):
    input_array: np.ndarray = input_image[slice_idx]
    input_array = np.uint8(input_array / np.max(input_array) * 255)
    return cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)


def prompt_based_prediction_sam_style_combination(predictor, prompt_points: list[int], prompts_bbox: list[list[int]],
                                                  score_filtering: bool = False, multimask_output: bool = False) -> \
        Optional[np.ndarray]:
    prediction: Optional[np.ndarray] = None
    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return prediction
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        pc: list[int] = prompt_points[:, :2]
        pl: int = prompt_points[:, -1]
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=box,
                                                        multimask_output=multimask_output)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        pc: list[int] = prompt_points[:, :2]
        pl: int = prompt_points[:, -1]
        prediction, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=None,
                                                  multimask_output=multimask_output)
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box,
                                                        multimask_output=multimask_output)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    # filter which mask to use -> default: take the first output
    if score_filtering and multimask_output:
        prediction = prediction[np.argmax(scores)]
    else:
        prediction = prediction[0]  # take the first output
    prediction = np.array(prediction > 0, dtype=np.uint8)
    return prediction


def prompt_based_prediction_sam_style_simple(predictor, prompt: list[int], score_filtering=False,
                                             multimask_output=False) -> Optional[np.ndarray]:
    preds: Optional[np.ndarray] = None
    if len(prompt) == 0:  # no prompt
        return preds
    if prompt.shape[-1] == 3:  # one or multiple points
        pc = prompt[:, :2]
        pl = prompt[:, -1]
        preds, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, multimask_output=multimask_output)
    else:  # bounding box
        if len(prompt.shape) == 1:  # one box without nested list
            preds, scores, _ = predictor.predict(box=prompt, multimask_output=multimask_output)
        else:  # box(es) with nested list
            for box in prompt:
                preds_single, scores, _ = predictor.predict(box=box, multimask_output=multimask_output)
                if preds is None:
                    preds = preds_single
                else:
                    preds += preds_single
    # filter which mask to use -> default: take the first output
    if score_filtering and multimask_output:
        preds = preds[np.argmax(scores)]
    else:
        preds = preds[0]
    preds = np.array(preds > 0, dtype=np.uint8)
    return preds
