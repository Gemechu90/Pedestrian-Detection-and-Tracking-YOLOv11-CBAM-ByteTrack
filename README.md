# 👥 People Detection & Multi-Object Tracking
### YOLO11m + CBAM Attention + ByteTrack

<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-People%20Detection-blue?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/YOLO11m-Object%20Detection-orange?style=for-the-badge" alt="YOLO11m">
  <img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-purple?style=for-the-badge" alt="ByteTrack">
  <img src="https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-8.4.137-green?style=for-the-badge" alt="Ultralytics">
</p>

<p align="center">
  <b>A complete computer-vision pipeline for detecting and tracking people in video using YOLO11m and ByteTrack, with a CBAM attention-module implementation.</b>
</p>

---

## ✨ Project at a Glance

This repository contains a notebook-based implementation for **person detection, video analysis, and multi-object tracking (MOT)**.

The workflow starts from the **Caltech Pedestrian Dataset**, converts its annotations into YOLO format, prepares a training/validation dataset, trains a YOLO11m detector, evaluates detection performance, and then applies **ByteTrack** to maintain consistent identities across video frames.

### 🔥 Core Pipeline

```text
Caltech Pedestrian Dataset
          │
          ▼
   Annotation Parsing
          │
          ▼
  YOLO-format Conversion
          │
          ▼
 Dataset Filtering & Sampling
          │
          ▼
      YOLO11m
  Person Detection
          │
          ▼
   Bounding Boxes + Scores
          │
          ▼
       ByteTrack
 Multi-Object Tracking
          │
          ▼
 Track IDs + Trajectories
          │
          ├──────────────► Annotated Video
          │
          └──────────────► tracker.txt
```

---

## 🎯 Objectives

The project is designed to:

- Detect **persons** in images and video.
- Convert Caltech pedestrian annotations into **YOLO bounding-box format**.
- Build dedicated training and validation subsets.
- Train a **YOLO11m** object detector.
- Evaluate detection using **Precision, Recall, mAP@50, and mAP@50–95**.
- Apply **ByteTrack** for persistent object identities across frames.
- Generate tracking results suitable for further MOT analysis.
- Measure video inference performance in FPS.
- Export frame-level tracking information to a text file.

---

## 🧠 Technologies

| Component | Technology |
|---|---|
| Programming | Python 3.10 |
| Deep Learning | PyTorch |
| Object Detection | YOLO11m |
| Attention Module | CBAM |
| Multi-Object Tracking | ByteTrack |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Progress Monitoring | tqdm |
| MOT Evaluation | motmetrics, TrackEval |
| Assignment / Environment | Jupyter / Kaggle |

---

## 📦 Dataset

The notebook uses the **Caltech Pedestrian Dataset** and focuses on the `person` class.

### Annotation processing

The notebook:

1. Reads Caltech `.vbb` annotation files.
2. Extracts pedestrian bounding boxes.
3. Keeps the `person` class.
4. Applies a pedestrian-height filter of **75–250 pixels**.
5. Converts bounding boxes from Caltech coordinates to normalized YOLO format.
6. Creates frame metadata in CSV format.
7. Generates image and label files for YOLO training.

### Dataset statistics recorded by the notebook

| Item | Value |
|---|---:|
| Ground-truth annotation boxes | **50,775** |
| Filtered train images | **18,041** |
| Filtered validation images | **12,025** |
| Validation images used in the recorded YOLO evaluation | **2,498** |
| Validation instances in that evaluation | **3,534** |
| Classes | **1 — person** |
| Image resolution used by YOLO | **640** |

> **Note:** The notebook samples/filteres the original dataset rather than using every available frame. The exact generated files depend on the execution environment and source dataset availability.

---

## 🏗️ Model Architecture

### YOLO11m Detector

The detector is initialized from the pretrained:

```text
yolo11m.pt
```

The training configuration recorded in the notebook includes:

```text
Epochs:       30
Batch size:   64
Image size:   640
Classes:      1
Class:        person
Optimizer:    auto
Device:       CUDA GPU
```

The recorded training environment used:

```text
Ultralytics 8.4.137
Python 3.10.10
PyTorch 2.0.0
CUDA GPU: Tesla P100-PCIE-16GB
```

### CBAM Attention

The notebook also implements a **Convolutional Block Attention Module (CBAM)** consisting of:

- Channel Attention
- Spatial Attention

The CBAM implementation is attached to selected YOLO11m backbone stages in the custom `YOLO11m_CBAM` class.

Conceptually:

```text
Input Feature Map
       │
       ▼
Channel Attention
       │
       ▼
Refined Feature Map
       │
       ▼
Spatial Attention
       │
       ▼
CBAM-Enhanced Features
```

### ⚠️ Reproducibility Note

The notebook contains both the **CBAM-enhanced model implementation** and a later training cell that explicitly initializes:

```python
model = YOLO("yolo11m.pt")
```

Therefore, the recorded evaluation result should be described as the **executed YOLO11m experiment**, while the CBAM implementation is present as a custom model component. If the research claim is specifically **“YOLO11m-CBAM”**, the training cell should be updated so that the CBAM-enhanced model itself is the model being trained and evaluated.

This distinction is important for reproducible scientific reporting.

---

## 🚀 Detection Performance

The recorded validation run reports:

| Metric | Recorded value |
|---|---:|
| Precision | **0.761** |
| Recall | **0.666** |
| mAP@50 | **0.702** |
| mAP@50–95 | **0.389** |

Recorded validation speed:

```text
Preprocess: 0.8 ms/image
Inference:  12.1 ms/image
Postprocess: 0.7 ms/image
```

The notebook also generates validation curves and metric visualizations for:

- Precision
- Recall
- mAP@50
- mAP@50–95
- Precision–Recall curves
- F1 curves
- Detection results

---

## 🎥 ByteTrack Multi-Object Tracking

After detection, the project uses **ByteTrack** to associate detections across consecutive frames.

The tracker configuration used in the notebook is:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
fuse_score: True
```

Tracking inference is configured with:

```text
confidence threshold: 0.25
IoU threshold:        0.50
persist:              True
```

### Tracking output

For each detected person, the system can retain:

- `track_id`
- bounding-box coordinates
- frame number
- tracking trajectory information

This converts frame-level detections into persistent identities.

---

## 📊 Example Tracking Flow

```text
Frame t
 ┌──────────────────────────────┐
 │ Person A → Detection         │
 │ Person B → Detection         │
 └──────────────────────────────┘
              │
              ▼
          ByteTrack
              │
              ▼
 ┌──────────────────────────────┐
 │ Person A → ID 1              │
 │ Person B → ID 2              │
 └──────────────────────────────┘
              │
              ▼
Frame t+1
 ┌──────────────────────────────┐
 │ Person A → ID 1              │
 │ Person B → ID 2              │
 │ Person C → ID 3              │
 └──────────────────────────────┘
```

The key idea is that **the same person can retain an identity across multiple frames**, rather than being treated as a new detection in every frame.

---

## ⚡ Video Processing

The notebook creates sample videos from selected frame segments and runs the trained detector/tracker on them.

The recorded sample tracking experiment processes:

```text
182 frames
Resolution: 640 × 480
```

The notebook also contains an FPS measurement procedure based on total processing time and the number of processed frames.

> FPS is hardware-, video-, model-, and execution-dependent. For reproducible reporting, benchmark it again on the target machine.

---

## 📁 Project Structure

A clean repository based on the notebook can use the following structure:

```text
people-detection-and-tracking-yolov11-bytetrack/
│
├── README.md
├── people-detection-and-tracking-yolov11-bytetrack.ipynb
│
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
│
├── csv_files/
│   ├── frame_metadata.csv
│   ├── train_frame_filtered.csv
│   └── val_frame_filtered.csv
│
├── sample_videos/
│   ├── sample_video_1.mp4
│   └── sample_video_2.mp4
│
├── output_videos/
│
├── custom_ultralytics/
│   └── nn/
│       └── modules/
│           └── cbam.py
│
├── bytetrack.yaml
├── custom_dataset.yaml
├── tracker.txt
│
└── runs/
    └── detect/
        ├── train/
        └── val/
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd people-detection-and-tracking-yolov11-bytetrack
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ultralytics
pip install motmetrics
pip install lap
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

Additional scientific packages used by the notebook include:

```bash
pip install numpy pandas scipy opencv-python matplotlib seaborn plotly tqdm
```

---

## ▶️ Running the Notebook

Open:

```text
people-detection-and-tracking-yolov11-bytetrack.ipynb
```

Then execute the notebook in the following order:

```text
1. Preparation
        ↓
2. Data Preprocessing
        ↓
3. Data Exploration
        ↓
4. YOLO11 Object Detection
        ↓
5. ByteTrack Object Tracking
        ↓
6. Tracking Evaluation / Export
```

For Kaggle execution, make sure the Caltech Pedestrian Dataset is attached and available at the path expected by the notebook.

---

## 🧪 Training

The notebook trains YOLO11m using:

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/kaggle/working/custom_dataset.yaml",
    epochs=30,
    batch=64,
    verbose=True
)
```

The dataset configuration contains:

```yaml
path: /kaggle/working/datasets
train: /kaggle/working/datasets/images/train
val: /kaggle/working/datasets/images/val

nc: 1

names:
  - person
```

---

## 🔎 Detection on Video

The notebook loads the trained weights from:

```text
runs/detect/train/weights/best.pt
```

and performs person detection with a confidence threshold of **0.50** in the dedicated detection function.

Conceptually:

```python
results = model.predict(video_or_frames)
```

The resulting detections can be represented as:

```text
bounding box + confidence
```

---

## 🧭 Tracking on Video

ByteTrack can be invoked through Ultralytics:

```python
results = model.track(
    source=video,
    tracker="bytetrack.yaml",
    persist=True,
    conf=0.25,
    iou=0.5,
    stream=True
)
```

The tracking stage produces persistent IDs and bounding boxes for detected people.

---

## 📄 Tracking File Format

The notebook exports tracking information to:

```text
tracker.txt
```

Each row follows the structure:

```text
frame_id, track_id, x, y, width, height, confidence, -1, -1, -1
```

Example:

```text
1,1,320.42,240.18,85.32,190.21,1,-1,-1,-1
```

This format is useful as an intermediate representation for MOT evaluation workflows.

---

## 📈 Evaluation

### Detection metrics

The notebook evaluates:

```text
Precision
Recall
mAP@50
mAP@50–95
```

### Tracking evaluation

The notebook installs tools intended for MOT evaluation:

```text
motmetrics
lap
TrackEval
```

These tools can be used for more advanced tracking metrics such as:

```text
MOTA
IDF1
HOTA
ID switches
False positives
False negatives
```

The attached notebook does not provide a complete recorded table of all of these tracking metrics, so they should be calculated separately before being reported as experimental results.

---

## 🌟 Key Features

- ✅ Person-only object detection
- ✅ YOLO11m pretrained initialization
- ✅ Custom Caltech-to-YOLO preprocessing
- ✅ Automated annotation conversion
- ✅ Dataset filtering and sampling
- ✅ CBAM attention-module implementation
- ✅ ByteTrack integration
- ✅ Persistent track IDs
- ✅ Video inference
- ✅ FPS measurement
- ✅ Tracking-result export
- ✅ Detection metric visualization
- ✅ MOT evaluation tooling
- ✅ Kaggle/Jupyter-compatible workflow

---

## 🔬 Research Value

This project provides a practical foundation for research in:

- Multi-Object Tracking (MOT)
- Intelligent Video Surveillance
- Pedestrian Detection
- Crowd Analysis
- Smart-City Video Analytics
- Traffic and Public-Space Monitoring
- Computer Vision
- Deep Learning
- Real-Time Video Understanding

The combination of **object detection + attention modeling + temporal identity association** creates a useful experimental framework for studying robust pedestrian perception in video.

---

## 🧩 Recommended Research Extensions

The current notebook can be extended with:

1. **True YOLO11m-CBAM training**
   - Train the CBAM-enhanced architecture directly.
   - Compare it against the baseline YOLO11m model.

2. **Ablation study**
   - YOLO11m
   - YOLO11m + CBAM
   - YOLO11m + ByteTrack
   - YOLO11m + CBAM + ByteTrack

3. **Tracking metrics**
   - MOTA
   - IDF1
   - HOTA
   - ID switches
   - FP/FN

4. **Runtime analysis**
   - FPS
   - latency
   - GPU memory
   - model size

5. **Robustness evaluation**
   - occlusion
   - crowd density
   - small pedestrians
   - illumination changes
   - camera motion

6. **Deployment**
   - NVIDIA Jetson
   - edge GPU
   - real-time surveillance camera
   - embedded computer-vision systems

---

## ⚠️ Reproducibility & Reporting Notes

For scientific or publication use, keep the following points explicit:

- The notebook title refers to **YOLOv11-CBAM + ByteTrack**.
- CBAM classes and a `YOLO11m_CBAM` model are implemented in the notebook.
- The recorded training cell, however, trains `YOLO("yolo11m.pt")`.
- Consequently, the reported **0.761 Precision / 0.666 Recall / 0.702 mAP@50 / 0.389 mAP@50–95** should not automatically be attributed to the CBAM-enhanced model without retraining that model.
- Tracking metrics such as MOTA, IDF1, and HOTA should be generated from ground-truth tracking annotations before being reported as final MOT results.
- FPS should be reported together with the hardware and inference configuration used.

This makes the repository easier to reproduce and prevents ambiguity between **implemented architecture** and **executed experiment**.

---

## 📚 References & Resources

- **Ultralytics YOLO:** https://github.com/ultralytics/ultralytics
- **ByteTrack:** https://github.com/ifzhang/ByteTrack
- **TrackEval:** https://github.com/JonathonLuiten/TrackEval
- **py-motmetrics:** https://github.com/cheind/py-motmetrics
- **Caltech Pedestrian Dataset:** https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

---

## 👤 Author

**Gemechu Geleta**

Lecturer & Computer Engineering Researcher  
MSc in Computer Engineering

Research interests include:

```text
Computer Vision • Deep Learning • Object Detection
Multi-Object Tracking • Artificial Intelligence
Intelligent Video Analytics • Edge AI
```

---

## 📜 License

If you publish this repository, add an explicit license such as **MIT**, **Apache-2.0**, or another license appropriate to your project and dataset/model usage.

> Dataset and pretrained-model usage should follow their respective licenses and terms.

---

## ⭐ Citation

If this repository contributes to your research, software, or experiments, please cite the repository using the citation information you provide in the final published version.

```bibtex
@software{gamachu_geleta_people_tracking,
  author  = {Gemechu Geleta},
  title   = {People Detection and Multi-Object Tracking using YOLO11m, CBAM and ByteTrack},
  year    = {2026},
  note    = {Computer Vision Research Project}
}
```

---

<p align="center">
  <b>👁️ Detect → 🎯 Associate → 🧭 Track → 📊 Evaluate</b>
</p>

<p align="center">
  <i>Building reliable video intelligence with modern computer vision.</i>
</p>
