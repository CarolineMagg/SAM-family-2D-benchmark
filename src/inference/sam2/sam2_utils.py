import sys
from pathlib import Path

from src.project_root import PROJECT_ROOT

SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "sam2"  # Get the absolute path to `submodules/sam2`
if SAM_MODULE_PATH not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from submodules.sam2.sam2.build_sam import build_sam2_video_predictor, build_sam2
from submodules.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

all_sam2_model_type_options = ["sam2_hiera_large", "sam2.0_hiera_large",
                               "sam2_hiera_tiny", "sam2.0_hiera_tiny",
                               "sam2_hiera_small", "sam2.0_hiera_small",
                               "sam2_hiera_base_plus", "sam2.0_hiera_base_plus",
                               "sam2_new_hiera_large", "sam2.1_hiera_large",
                               "sam2_new_hiera_tiny", "sam2.1_hiera_tiny",
                               "sam2_new_hiera_small", "sam2.1_hiera_small",
                               "sam2_new_hiera_base_plus", "sam2.1_hiera_base_plus"]

import hydra
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
    sam2_config_paths = (PROJECT_ROOT / "submodules" / "sam2" / "sam2" / "configs")
    hydra.initialize_config_dir(config_dir=str(sam2_config_paths), version_base="1.2")
    cfg = hydra.compose(config_name="sam2.1/sam2.1_hiera_t.yaml")


def create_sam2_predictor(device, model_type, video_predictor=False):
    checkpoint_folder = PROJECT_ROOT / "checkpoints" / "sam2"
    if model_type == "sam2_hiera_large" or model_type == "sam2.0_hiera_large":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_large.pt"
        model_cfg = "sam2/sam2_hiera_l.yaml"
    elif model_type == "sam2_hiera_base_plus" or model_type == "sam2.0_hiera_base_plus":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_base_plus.pt"
        model_cfg = "sam2/sam2_hiera_b+.yaml"
    elif model_type == "sam2_hiera_tiny" or model_type == "sam2.0_hiera_tiny":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_tiny.pt"
        model_cfg = "sam2/sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small" or model_type == "sam2.0_hiera_small":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_small.pt"
        model_cfg = "sam2/sam2_hiera_s.yaml"
    elif model_type == "sam2_new_hiera_large" or model_type == "sam2.1_hiera_large":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_large.pt"
        model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
    elif model_type == "sam2_new_hiera_base_plus" or model_type == "sam2.1_hiera_base_plus":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_base_plus.pt"
        model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
    elif model_type == "sam2_new_hiera_tiny" or model_type == "sam2.1_hiera_tiny":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_tiny.pt"
        model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
    elif model_type == "sam2_new_hiera_small" or model_type == "sam2.1_hiera_small":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_small.pt"
        model_cfg = "sam2.1/sam2.1_hiera_s.yaml"
    else:
        raise ValueError(f"invalid model_type {model_type}")
    if video_predictor:
        predictor = build_sam2_video_predictor(str(model_cfg), str(sam2_checkpoint), device=device)
    else:
        sam2_model = build_sam2(str(model_cfg), str(sam2_checkpoint), device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    return predictor
