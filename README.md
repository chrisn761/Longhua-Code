# Explainable Multi-Perspective Posture Assessment

This repository provides the configuration files and core scripts used for the visual-model development and assessment components of the multi-perspective posture assessment framework.

## Repository contents

| File | Description |
| --- | --- |
| `train-pose.py` | Training script for the anatomical key-point localization model. |
| `segmenttrain-yolov12.py` | Training script for the regional segmentation model. |
| `HumanBack.yaml` | Dataset configuration for back-view key-point localization. |
| `HumanSide_10.yaml` | Dataset configuration for one side-view key-point localization setting. |
| `HumanSide_253_handscover.yaml` | Dataset configuration for the side-view, arms-crossed key-point localization setting. |
| `overhead_12.yaml` | Dataset configuration for top-view key-point localization. |
| `segment-yolo12.yaml` | Dataset configuration for regional segmentation. |

## Data availability

The original image datasets and clinical annotations are not included in this repository. They contain sensitive clinical information and are large in size. Access may be requested from the corresponding author for reasonable research purposes, subject to institutional ethics approval and participant-privacy requirements.

## Notes

The YAML files define the corresponding dataset settings and data paths. After authorized access to the data is obtained, users should update the dataset paths in these files to match their local environment.

## Requirements

The visual-model scripts are based on the Ultralytics YOLO framework. Please configure the required environment and update local file paths before execution.
