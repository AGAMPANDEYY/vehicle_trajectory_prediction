# Vehicle Trajectory Prediction using Aerial Images and Object Detection
![{D1BDD3E3-E4F7-49D4-9094-E119D205E330}](https://github.com/user-attachments/assets/ec57b77b-fe41-4d5c-b9d9-6a96848dd880)

## Overview

This project presents a unified pipeline for multivariate vehicle trajectory prediction and conflict zone detection using aerial drone footage. Leveraging deep learning models, attention mechanisms, and physics-based filtering, the system predicts future vehicle positions and estimates road safety using Post-Encroachment Time (PET) metrics.

Developed as part of the CEN-300 practical coursework, the work integrates multiple components such as custom object detection, spatiotemporal trajectory modeling, and conflict risk analytics.

**Repository:** [https://github.com/AGAMPANDEYY/vehicle_trajectory_prediction](https://github.com/AGAMPANDEYY/vehicle_trajectory_prediction)

---

## Problem Statement

Given a 5-minute aerial video captured by a UAV, the objective is to:

- Detect and track all vehicles (including two-wheelers).
- Predict future trajectories using motion context and neighboring vehicle interactions.
- Estimate potential conflict zones based on predicted paths and PET metrics.

---

## Project Workflow

1. **Data Collection and Annotation**  
   - Aerial footage from drones at ~30m altitude.
   - Manual annotation of 250+ images with three classes: Car, Bike, Bus.
   - Augmentation techniques applied: noise, brightness, cropping, rotation.

2. **Vehicle Detection & Tracking**  
   - **Model:** YOLOv8s for object detection, ByteTrack for multi-object tracking.  
   - **Libraries:** Roboflow Supervision, OpenCV, NumPy.  
   - Output: Vehicle bounding boxes and trajectories (frame-wise tracking).

3. **Trajectory Prediction Models**  
   - **Baseline:** Kalman Filter for trajectory smoothing and velocity estimation.  
   - **Deep Learning Model:**  
     - LSTM-based motion model.  
     - SDT-ATT (Spatial–Dynamic Attention with Bi-LSTM) for context-aware prediction.  
     - Incorporates 3 neighboring vehicles, bi-directional memory, and attention.

4. **Conflict Zone Detection**  
   - Conflict zones identified using Post-Encroachment Time (PET):  
     - PET = entry time of following vehicle – exit time of leading vehicle.  
     - Risk levels categorized from **SAFE** to **HIGH RISK** based on PET thresholds.  
   - Total of 141 conflict zones analyzed and mapped.
     
![{4A16A367-E9E6-48F7-8D39-25FA4724DEB2}](https://github.com/user-attachments/assets/15331ef3-ba20-4ab4-9ac2-bf925e2ca9b6)
![{6282BAB8-CCC7-4AE5-A007-F8CA8D06B107}](https://github.com/user-attachments/assets/7cfb8997-1c49-4446-9536-e4f5aeb6c5a2)

---

## Technical Summary

### Libraries & Tools

- **Computer Vision:** YOLOv8s, Roboflow, OpenCV
- **Tracking:** ByteTrack
- **Data Handling:** Pandas, NumPy
- **Modeling:** PyTorch, TensorFlow
- **Visualization:** Matplotlib, Seaborn

### Architecture

- **Detection:** YOLOv8s + ByteTrack
- **Trajectory Prediction:**  
  - Kalman Filter  
  - LSTM  
  - SDT-ATT with attention mechanism and Bi-LSTM layers
- **Conflict Zone Analysis:**  
  - PET computation  
  - Heatmap and statistical distribution analysis
![{ACA9A4F9-E10D-4914-A70C-016E17C9D1B3}](https://github.com/user-attachments/assets/c4d3ced8-1275-49ba-9126-a1c70ed0b8ca)

---

## Evaluation Metrics

- **ADE (Average Displacement Error)**  
- **FDE (Final Displacement Error)**  
- **PET Risk Categorization:**
  - PET ≤ 0: Near Miss
  - 0 < PET ≤ 1.5s: High Risk
  - 1.5 < PET ≤ 3s: Moderate Risk
  - 3 < PET ≤ 5s: Low Risk
  - PET > 5s: Safe

---

## Results

- SDT-ATT outperforms traditional LSTM and Kalman approaches in multivariate prediction tasks.
- ~25–30% of observed vehicle interactions occurred in risk-prone PET windows.
- PET ≤ 5s recorded in ~40–45% of cases.
- Risk category heatmaps generated for conflict zone intensity visualization.

---

## Limitations

- SDT-ATT assumes perfect visibility of neighboring vehicles; occlusions affect accuracy.
- PET analysis relies purely on positional data without considering lane semantics.
- Lane-change behaviors are inferred, not explicitly modeled.
- Model performance is sensitive to tracking accuracy and annotation quality.

---

## Future Work

- Incorporate semantic maps including road boundaries and lane markings.
- Introduce explicit lane-change classification using multi-task learning.
- Deploy on live UAV feeds for real-time conflict detection.
- Extend prediction to 3D space (elevation-aware models).
- Enhance robustness using sensor fusion (e.g., LiDAR) and Transformer-based models.

---

## Authors

- Agam Pandey (22113009)  
- Hardik Chawla (22113056)  
- Krish Sharma (22124021)  
- Samarth Pratap Singh (22113129)

---

## References

- [Vehicle Trajectory Prediction Based on Multivariate Interaction Modeling (SDT-ATT)](https://ieeexplore.ieee.org/document/10323306/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)


ByteTrack: https://github.com/ifzhang/ByteTrack
---
