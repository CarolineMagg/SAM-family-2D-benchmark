# SAM-family-2D-benchmark

This is the repository for the MIDL 2025 full
paper [Zero-shot capability of 2D SAM-family models for bone segmentation in CT scans](https://openreview.net/forum?id=AUv6NhK9aH).

### SAM-family models

SAM-family models are segmentation foundation model (FMs), which work with input prompts to identify the object of
interest (i.e., bounding box, points/clicks).
The following SAM-family models are included in this repository:

* SAM
* SAM2 Imagepredictor
* Med-SAM
* SAM-Med2D

### Prompting strategies

The supported prompts are static and are automatically generated from the reference mask.
A prompt consists of at least one primitive (i.e., (a) bounding box, (b) center, (c) centroid, (e) positive or (f)
negative random points) and one component selection criteria (i.e., largest or up to 5 components).

![visualization of prompting strategies](/assets/prompt_strategies.PNG)

## Installation

### Dependencies

1. Download repository
2. Create an environment with the following dependencies:

* torch >= 2.5.0
* nibabel
* numpy
* pandas
* natsort
* scikit-image
* opencv-python
* tqdm
* plotly
* albumentations
* hydra-core
* iopath

### Download checkpoints

As of now, all checkpoints have to be manually downloaded from the origin sources:

* SAM: [all](https://github.com/facebookresearch/segment-anything/tree/6fdee8f2727f4506cfbbe553e23b895e27956588)
* Med-SAM: [medsam_vit_b](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)
* SAM-Med2D: [sam-med2d_b](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view)
* SAM2: [all](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)

Place the files in the folder structure:

````
project_folder/
└── checkpoints/
    ├── sam/
    ├── sam2/
    ├── med-sam/
    └── sam-med2d/
└── src /
└── submodules /
````

### SAM-family submodules

The official github repositories of all SAM-family models are setup as submodules at pre-defined commits:

* [SAM](https://github.com/facebookresearch/segment-anything/tree/6fdee8f2727f4506cfbbe553e23b895e27956588)
* [Med-SAM](https://github.com/bowang-lab/MedSAM/tree/2b7c64cf80bf1aba546627db9b13db045dd1cbab)
* [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/bfd2b93b1158100c8abd81f61766a2de92c1c175)
* [SAM2](https://github.com/facebookresearch/sam2/tree/29267c8e3965bb7744a436ab7db555718d20391a)

## Usage

### Dataset Preparation

We recommend to use
the [nnUNet format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to store data.


### Prompt generation

The inference works with offline generated json prompt files which stores the prompts in a nested dictionary; storing
information such as file id, slice id and prompt type. 
````
# set random seed to get reproducible prompts
np.random.seed(42)
# generate prompts
path_labels = Path(<path_to_labels>)
json_file = Path(<json_prompt_file_name>)
labels = [1, 2]
all_2d_prompts = generate_2d_prompts_for_folder(path_labels, labels)
all_2d_prompts["image_path"] = <path_to_images>
# write file
with open(json_file, "w") as f:
    json.dump(all_2d_prompts, f)
print(f"write {json_file}")
````

### Inference

The inference can be started by calling the inference method of the models, e.g.:
```
json_file = Path(<json_prompt_file_name>)
output_folder = Path(<path_to_output_folder>)
prompt_type = ["center"]  # ["bbox", "center"]
number_prompts = 1  # 5
random_number_prompts = 1  # up to 10
model_type = "vit_b"  # depends on the model options

run_inference_sam(json_file, output_folder, 
                  prompt_type, number_prompts, 
                  random_number_prompts, model_type)
```