# AeroSight — High-Precision Real-Time Aircraft Detection

## Overview
Perception for aerial systems such as, collision avoidance, autonomous navigation, or airspace awareness, remains one of the most challenging domains in computer vision. I built **AeroSight** to deepen my practical experience in real-time object detection, tracking, and perception pipeline design.

The goal was to design something that reflects how real perception stacks are built in industry: trained models, exported for deployment, integrated into real time C++ code, and paired with classical state estimation.

This project emphasizes:
- End-to-end perception pipeline design
- Real-time constraints and performance tradeoffs
- Integration of deep learning with classical filtering techniques
- Modular C++ implementations suitable for robotics and aerospace systems



## Gallery
<table width="100%">
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/5f3cc43c-241e-4e0d-aece-9767d861a342" alt="Image 1" style="width:100%;">
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/50672d5c-40d7-4d3b-bda9-e308443eaaeb" alt="Screenshot 2025-12-25 222329" style="width:100%;">
    </td>
  </tr>
</table>

https://github.com/user-attachments/assets/7aa35610-11c9-4060-a0f4-65407c4838d4

## Tech Stack

| Technology | Role |
|---------|------|
| **PyTorch / Ultralytics** | Model training and experimentation |
| **Python** | Training, preprocessing, and tooling |
| **ONNX Runtime** | Inference in C++ |
| **OpenCV** | Image processing, visualization, video I/O |



## Technical Overview

#### Object Detection — YOLOv11n
The detection model (Ultralytics YOLOv11n base) was trained (Using a Google Colab Nvidia A100 GPU instance) on a curated aircraft dataset from Kaggle (https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data). The training process included:
- Dataset preprocessing and labeling
- Image augmentation to improve generalization
- Hyperparameter tuning for small-object detection
- Validation and export to ONNX

#### Inference & Runtime — C++

- **`main.cpp`**
  - Application entry point
  - Video stream handling
  - Visualization and runtime loop

- **`detector.cpp / detector.h`**
  - ONNX Runtime integration
  - Image preprocessing and normalization
  - Post-processing (confidence filtering, bounding boxes)

- **`tracker.cpp / tracker.h`**
  - State estimation and temporal tracking
  - Interfaces cleanly with detection outputs

#### Object Tracking — Kalman Filter

To maintain consistent object identities across frames, AeroSight integrates a Kalman filter-based tracker.
The Kalman filter provides:
- Predictive motion modeling
- Noise-aware measurement fusion
- Temporal smoothing of bounding boxes

#### Repository Layout

- `AeroTrack/`
  - `main.cpp` – video inference pipeline and UI overlay logic.
  - `detector.cpp / detector.h` – ONNX Runtime, OpenCV wrapper around the YOLOv11 ONNX model.
  - `tracker.cpp / tracker.h` – tracking hooks, extension point for multi‑frame tracking Kalman filter.
  - `CMakeLists.txt`
- `train_yolov11.ipynb` – notebook for training and exporting the YOLOv11 model using Google Colab.
- `military-aircraft-yolo/` – dataset configuration and labels.
- `requirements.txt`
- `.gitignore`

### Quickstart

```powershell
cmake --build . --config Release

./build/Release/AeroTrack.exe
```
