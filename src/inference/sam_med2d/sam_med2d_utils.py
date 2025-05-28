import argparse
import torch
from src.project_root import PROJECT_ROOT
from submodules.SAMMed2D.segment_anything import sam_model_registry
from submodules.SAMMed2D.segment_anything.predictor_sammed import SammedPredictor


def create_sammed2d_predictor(device: torch.device, model_type: str = "vit_b"):
    args_sam = argparse.Namespace()
    args_sam.image_size = 256
    args_sam.encoder_adapter = True  # adapter layer in encoder
    args_sam.sam_checkpoint = PROJECT_ROOT / "checkpoints" / "sam_med2d" / "sam-med2d_b.pth"
    model = sam_model_registry[model_type](args_sam).to(device)
    return SammedPredictor(model)
