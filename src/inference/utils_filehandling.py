from typing import Union

import numpy as np
import nibabel as nib
from pathlib import Path


def read_image_depth_first(path_image: Path, file_id: str, file_ending: str = ".nii.gz"):
    nii_image, nii_image_affine = read_nii_file(path_image, file_id, file_ending)
    input_image = np.asarray(nii_image.dataobj)
    input_image = np.moveaxis(input_image, -1, 0)  # depth is on first place
    return input_image, nii_image_affine


def read_nii_file(path_image: Path, file_id: str, file_ending: str = ".nii.gz"):
    img_file_name = path_image / (file_id + "_0000" + file_ending)
    nii_image = nib.load(img_file_name)
    return nii_image, nii_image.affine


def write_output_masks_to_nii(output_msk: Union[list[np.ndarray], np.array], labels_lookup: dict, output_dir: Path,
                              file_id: str, file_ending: str, affine: np.ndarray, verbose: bool = False):
    for idx, msk in enumerate(output_msk):
        msk_ = np.moveaxis(msk, 0, -1)  # H, W, D
        label_name = list(labels_lookup.keys())[list(labels_lookup.values()).index(str(idx + 1))]
        segm_file_name = output_dir / label_name / (file_id + file_ending)
        nii = nib.Nifti1Image(msk_, affine=affine)
        nib.save(nii, segm_file_name)
        if verbose:
            print(f"write {segm_file_name}")


def write_output_mask_to_nii_simple(output_mask: np.ndarray, label_name: str, output_dir: Path,
                                    file_id: str, file_ending: str, affine: np.ndarray, verbose:bool =False):
    segm_file_name = output_dir / label_name / (file_id + file_ending)
    nii = nib.Nifti1Image(output_mask, affine=affine)
    nib.save(nii, segm_file_name)
    if verbose:
        print(f"write {segm_file_name}")
