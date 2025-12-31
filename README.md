# AeroSight — High-Precision Real-Time Aircraft Detection

## Overview
Perception for aerial systems such as, collision avoidance, autonomous navigation, or airspace awareness, remains one of the most challenging domains in computer vision. I built **AeroSight** to deepen my practical experience in real-time object detection, tracking, and perception pipeline design.

The goal was to design something that reflects how real perception stacks are built in industry: trained models, exported for deployment, integrated into real time C++ code, and paired with classical state estimation.

My goal was to design a system that closely mirrored how perception systems are built in industry. Including model training, running real time inference with hardware limitations, and fusing the results with classical state estimation.

This project implements the perception and tracking components commonly used in Sense-and-Avoid systems:
  - **Model Fine-Tuning**: Custom fine-tuning of the YOLO11n model on domain-specific aerial imagery to improve detection performance and robustness.
  - **Detection**: Real-time aircraft detection using optimized YOLO11n inference deployed via ONNX Runtime.
  - **Tracking**: Kalman Filter–based state estimation to maintain stable target tracks under noise and occlusion.
  - **Systems**: A modular C++ framework designed for low-latency execution and deployment on edge hardware.

## Gallery
<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/f1b949b0-ea76-4cd9-bb21-5e6bdc734941" style="width:100%;" />
      <br/>
      <em>Image shows Kalman-Only Constant-Velocity Tracker (KO-CV), indicated by red bounding box</em>
    </td>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/decd934e-b119-4eed-869c-84a3952d4833" style="width:100%;" />
      <br/>
      <em>Image shows multiclass (aircraft type) detection capabilities</em>
    </td>
  </tr>
</table>

https://github.com/user-attachments/assets/6b24323e-63bc-4192-aacc-7d9f81f6dd45

Video showing the detector being turned off after three seconds, allowing KO-CV to take over and predict motion. The detector is then re-enabled, and the track is successfully regained.

## Tech Stack

| Technology | Role |
|---------|------|
| **ONNX Runtime (CUDA)** | High-performance GPU-accelerated inference within the C++ pipeline |
| **OpenCV** | Image preprocessing, visualization, and video I/O in the runtime system |
| **PyTorch / Ultralytics** | Fine-tuning the YOLOv11n model and exporting trained weights |
| **Python** | Dataset preprocessing, training orchestration, evaluation, and tooling |

## Deep Dive: The Perception Pipeline

#### Object Detection (YOLO11)

The system utilizes a finetuned **YOLO11n** model, trained (using an Nvidia A100 GPU instance through Google Colab) on a specialized military aircraft dataset from Kaggle.
- **Preprocessing:** Frames are resized to $640 \times 640$ using **letterboxing** to preserve aspect ratios, followed by channel-swapping (BGR to RGB) and pixel normalization.
- **Post-Processing:** Implements **Non-Maximum Suppression (NMS)** in C++ to prune redundant bounding boxes. Detections are filtered based on a configurable confidence threshold ($T_{conf} > 0.45$).

#### Temporal Tracking (Kalman Filter)

To solve the problem of intermittent detections (flicker), sensor noise, or possible object occlusion behing a cloud or terrain, AeroSight employs a **Linear Kalman Filter (KF)** for each tracked object.
- **State Vector:** The filter tracks an 8-dimensional state vector:
  $$x = [c_x, c_y, a, h, \dot{c}_x, \dot{c}_y, \dot{a}, \dot{h}]^T$$
  where $(c_x, c_y)$ is the box center, $a$ is the aspect ratio, $h$ is the height, and the remaining terms represent their respective velocities.
- **Motion Model:** A Constant-Velocity (CV) model predicts the aircraft's position in the next frame.
- **Robustness:** If the detector fails to track the aircraft, the KF enters **Predict-Only mode**, using its internal motion model to maintain the track until the detector re-acquires the target.

#### C++ Inference Engine
The core logic is implemented in C++ to ensure deterministic performance, maintian minimal overhead, and improve processing speeds.
- **Memory Management:** Utilizes smart pointers and pre-allocated tensors to minimize heap allocations during the inference loop.
- **Modularity:** The `Detector` class is decoupled from the `Tracker` class, allowing for "plug-and-play" swapping of models.

#### Repository Layout

- `AeroTrack/`
  - `main.cpp` – video inference pipeline and UI overlay logic.
  - `detector.cpp / detector.h` – ONNX Runtime, OpenCV wrapper around the YOLOv11 ONNX model.
  - `tracker.cpp / tracker.h` – tracking hooks, extension point for multi‑frame tracking Kalman filter.
- `train_yolov11.ipynb` – notebook for training and exporting the YOLOv11 model using Google Colab.
- `requirements.txt`

### Quickstart

```powershell
cmake --build . --config Release

./build/Release/AeroTrack.exe
```
