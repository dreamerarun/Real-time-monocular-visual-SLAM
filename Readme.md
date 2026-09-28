# 🧭 Real-Time Visual SLAM — ORB-SLAM3 Style

A real-time **monocular Visual SLAM** project using a webcam to estimate camera motion, track visual observations, build a persistent 3D map, and visualize the camera trajectory in real time.

The project follows an **ORB-SLAM3-style visualization and workflow**, while the actual visual tracking backend is **NVIDIA cuVSLAM**.

> **Note:** This project is not the original ORB-SLAM3 implementation. It uses NVIDIA cuVSLAM for visual odometry/SLAM tracking and provides an ORB-SLAM3-inspired real-time visualization interface.

## Quick-Start Prerequisites

Before running the project, make sure the following are installed and available:

### Hardware

* A working **USB/integrated webcam**
* NVIDIA GPU recommended for cuVSLAM acceleration
* At least **8 GB RAM** recommended
* Display capable of running the Open3D 3D viewer

### Software

* **Python 3.8+**
* **OpenCV**
* **NumPy**
* **NVIDIA cuVSLAM**
* **Open3D** — required for the 3D visualization mode
* NVIDIA GPU drivers with CUDA support compatible with your cuVSLAM installation

### Python Dependencies

Install the required Python packages:

```bash
pip install numpy opencv-python open3d
```

> **Note:** `cuvslam` is an NVIDIA-provided dependency and must be installed/configured separately according to your NVIDIA cuVSLAM environment.

### Verify Your Webcam

Check that your camera is detected before starting SLAM:

```bash
python -c "import cv2; cap=cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"
```

If the output is:

```text
Camera: True
```

the default camera (`--camera 0`) is available.

### Verify NVIDIA GPU

Check that your NVIDIA GPU and driver are detected:

```bash
nvidia-smi
```

The project initializes cuVSLAM and performs GPU warm-up before tracking begins.

### Recommended Setup

For the complete experience, use:

```text
Webcam
   ↓
OpenCV
   ↓
NVIDIA cuVSLAM
   ↓
Visual Odometry / SLAM
   ↓
Pose + Observations + Landmarks
   ├──→ 2D Trajectory
   ├──→ Open3D 3D Map
   └──→ Saved SLAM Map
```

### Important

This project uses **NVIDIA cuVSLAM** as its SLAM backend. The repository is inspired by the concepts and visualization style of **ORB-SLAM3**, but it is **not the official ORB-SLAM3 implementation**.

---

## 🚀 Features

* 🎥 Real-time webcam input
* 🧭 Monocular visual tracking
* 📍 Real-time camera pose estimation
* 🗺️ Persistent 3D map points
* 📈 Camera trajectory visualization
* 🧊 Interactive Open3D 3D viewer
* 📷 Camera pose frustums
* 🎯 Visual feature observation overlay
* 🔴 Current camera position indicator
* 🔄 Follow-camera mode
* 💾 Save and load SLAM maps
* 📦 Export point cloud to `.ply`
* 📊 Real-time FPS display
* 🖥️ GPU-accelerated tracking through cuVSLAM
* 🔧 Configurable camera, resolution, FPS, and map loading

The project also includes fixes for coordinate orientation and interactive 3D visualization.

---

# 🧠 System Overview

The processing pipeline is:

```text
                 Webcam
                    │
                    ▼
             Camera Frame
                    │
                    ▼
              Grayscale
                    │
                    ▼
             NVIDIA cuVSLAM
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     Camera Pose          Observations
          │                   │
          │                   ▼
          │              Map Points
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
             SLAM Visualization
             ┌──────┴──────┐
             │             │
             ▼             ▼
        OpenCV View     Open3D View
             │             │
             ▼             ▼
       2D Trajectory     3D Map
```

---

# 🔍 How It Works

## 1. Webcam Input

The system captures frames directly from a webcam using OpenCV.

Default configuration:

```text
Camera ID: 0
Resolution: 640 × 480
Target FPS: 30
```

The camera parameters can be modified through command-line arguments.

The camera initialization and resolution configuration are implemented directly in the `WebcamSLAM` class.

---

## 2. Camera Model

For a general webcam, approximate intrinsic parameters are generated from the image resolution.

```text
fx = width × 0.8
fy = height × 0.8
cx = width / 2
cy = height / 2
```

The resulting camera matrix is:

```text
        ┌             ┐
        │ fx   0   cx │
K   =   │ 0   fy   cy │
        │ 0    0    1 │
        └             ┘
```

The current implementation assumes zero distortion for the webcam.

---

# ⚡ NVIDIA cuVSLAM

The visual tracking backend is **NVIDIA cuVSLAM**.

The tracker is configured for **monocular odometry**:

```python
cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
```

The implementation also enables observation export and final landmark export for visualization:

```python
cfg.enable_observations_export = True
cfg.enable_final_landmarks_export = True
```

The GPU is warmed up before tracker initialization.

---

# 📍 Camera Tracking

Every captured frame is converted to grayscale and passed to the SLAM tracker.

```text
Webcam Frame
     ↓
Grayscale Image
     ↓
cuVSLAM Tracker
     ↓
Camera Pose
     ↓
Trajectory
```

The estimated pose provides the camera translation:

```text
X
Y
Z
```

The position is continuously added to the camera trajectory when sufficient movement is detected.

---

# 🎯 Visual Feature Tracking

Tracked observations are displayed directly on the camera feed.

The visualization distinguishes between:

* Tracked landmarks
* Newly observed features

The feature count is also displayed in real time.

```text
Camera Feed
│
├── SLAM Status
├── Camera Position
├── Feature Points
└── FPS
```

The implementation retrieves observations from cuVSLAM and overlays them using OpenCV.

---

# 🗺️ 2D Trajectory Map

A separate OpenCV window displays a top-down trajectory representation.

The visualization uses:

```text
X → Horizontal movement
Z → Depth movement
Y → Height
```

The trajectory is plotted continuously as the camera moves.

The display also provides:

* Number of trajectory points
* Estimated traveled distance
* Current camera position

---

# 🌐 Interactive 3D SLAM Viewer

If Open3D is available, the project launches an interactive 3D viewer.

The viewer contains:

```text
                 3D SLAM MAP

        • • • • • • • • •
      •                  •
    •        MAP          •
   •       POINTS         •
    •                    •
      • • • • • • • • •
             │
             ▼
        Camera Path
```

The viewer includes:

* Coordinate frame
* Persistent map points
* Camera trajectory
* Current camera position
* Camera frustums
* Reference grid
* Interactive camera controls

The viewer is initialized with a coordinate frame, persistent point cloud, trajectory line set, camera indicator, and grid.

---

# 🧊 Open3D Controls

The 3D viewer supports interactive navigation.

| Control            | Action               |
| ------------------ | -------------------- |
| Left Mouse + Drag  | Rotate               |
| Right Mouse + Drag | Pan                  |
| Mouse Wheel        | Zoom                 |
| `F`                | Toggle Follow Camera |
| `R`                | Reset View           |

The viewer uses non-blocking Open3D updates so that interaction can continue while SLAM is running.

---

# 📌 Persistent Map Points

Unlike a visualization that only displays the latest frame, this project maintains map points across frames.

Map points are stored using landmark IDs:

```python
self.all_map_points = {}
```

This allows previously observed landmarks to remain visible as the camera moves.

The map can therefore progressively grow during a SLAM session.

---

# 📷 Camera Frustums

Camera poses can also be represented using 3D camera frustums.

The system keeps a limited number of recent camera poses:

```text
Maximum frustums = 30
```

This provides a visual representation of how the camera moved through the environment.

---

# 💾 Map Saving

The project supports saving the SLAM state.

Press:

```text
S
```

to save:

```text
slam_frame_<timestamp>.png
slam_data_<timestamp>.pkl
slam_data_<timestamp>.ply
```

The `.pkl` file stores:

* Camera trajectory
* Map points
* Frame count

The `.ply` file contains the point-cloud representation of the map.

---

# 📂 Map Loading

A previously saved SLAM map can be loaded when starting the program.

Example:

```bash
python slam.py --load-map slam_data_20260928_120000.pkl
```

The stored trajectory and map points are restored into the visualization.

---

# ⌨️ Runtime Controls

While SLAM is running:

| Key   | Function                  |
| ----- | ------------------------- |
| `Q`   | Quit                      |
| `ESC` | Quit                      |
| `S`   | Save frame and map        |
| `R`   | Reset SLAM                |
| `F`   | Toggle follow-camera mode |
| `V`   | Reset 3D view             |
| `L`   | Load saved map            |
| `C`   | Clear map points          |

These controls are implemented in the main SLAM loop.

---

# 📦 Dependencies

The implementation imports:

```text
Python
OpenCV
NumPy
NVIDIA cuVSLAM
Open3D
```

Main Python imports include:

```python
import cv2
import numpy as np
import cuvslam
import open3d
```

Open3D is optional. If it is not installed, the SLAM system can still operate without the 3D viewer.

---

# 🔧 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

---

## 2. Create a Python Environment

For example:

```bash
python3 -m venv slam_env
source slam_env/bin/activate
```

---

## 3. Install Python Dependencies

```bash
pip install numpy opencv-python open3d
```

Install **NVIDIA cuVSLAM** according to the NVIDIA/CUDA environment being used.

> cuVSLAM is a GPU-accelerated component and requires a compatible NVIDIA environment.

---

# ▶️ Running the Project

Run with the default webcam:

```bash
python slam.py
```

Default settings:

```text
Camera: 0
Width: 640
Height: 480
FPS: 30
3D Viewer: Enabled
```

---

# 🎥 Custom Camera

Specify a different camera:

```bash
python slam.py --camera 1
```

---

# 🖥️ Custom Resolution

```bash
python slam.py --width 1280 --height 720
```

---

# ⚡ Custom FPS

```bash
python slam.py --fps 30
```

---

# 🧊 Disable 3D Viewer

If Open3D is not required:

```bash
python slam.py --no-3d
```

This allows the system to run with the OpenCV camera and trajectory visualization without the interactive Open3D viewer.

The command-line options are defined in the program's `argparse` configuration.

---

# 🧪 Example

Start the system:

```bash
python slam.py
```

Then slowly move the camera around an environment.

The system will display:

```text
┌───────────────────────────────┐
│       Camera Feed             │
│                               │
│   •  •   •  •  •             │
│       Features                │
│                               │
│   Status: TRACKING            │
│   Position: X Y Z             │
│   Features: XXXX              │
│                               │
│                         FPS   │
└───────────────────────────────┘
```

At the same time, the trajectory is displayed in a separate window and the 3D viewer progressively builds the map.

---

# 📊 SLAM Visualization

The project provides three main visualization outputs:

### 1. Camera Feed

Shows:

* Tracking status
* Camera position
* Feature observations
* Feature count
* FPS

### 2. Top-Down Trajectory

Shows:

* Camera movement
* X-Z trajectory
* Current position
* Travel distance

### 3. Interactive 3D Map

Shows:

* Persistent map points
* Camera trajectory
* Current camera
* Camera frustums
* Coordinate frame
* Reference grid

---

# 🧮 Map Representation

The system maintains:

```text
Trajectory
    ↓
Camera poses over time

Map Points
    ↓
3D landmarks observed by the tracker

Camera Frustums
    ↓
Selected historical camera poses
```

The 3D map is updated periodically from the tracker landmarks.

---

# 🔄 SLAM Reset

Press:

```text
R
```

to reset the SLAM session.

The trajectory and map visualization are cleared and the tracker is initialized again.

---

# 🧹 Clear Map

Press:

```text
C
```

to clear the currently displayed map points while keeping the SLAM system running.

---

# 📈 Performance Considerations

For better tracking:

* Move the camera slowly.
* Use a well-lit environment.
* Avoid excessive motion blur.
* Provide textured surfaces.
* Avoid pointing the camera at large textureless regions.
* Maintain sufficient visual features.
* Use a camera with reasonably stable exposure.

The program itself recommends moving the camera slowly for better results.

---

# ⚠️ Important Limitations

This project uses **approximate webcam calibration** rather than a calibrated camera model.

The focal lengths are estimated from image dimensions:

```text
fx = width × 0.8
fy = height × 0.8
```

Therefore, the resulting metric accuracy can differ from a system using properly calibrated camera intrinsics.

The implementation is also a **monocular** configuration:

```text
OdometryMode.Mono
```

so it does not use stereo or RGB-D input.

---

# 🆚 ORB-SLAM3 vs This Project

| Feature                      | ORB-SLAM3                        | This Project            |
| ---------------------------- | -------------------------------- | ----------------------- |
| Visual SLAM                  | ✅                                | ✅                       |
| Monocular mode               | ✅                                | ✅                       |
| Stereo                       | ✅                                | ❌                       |
| RGB-D                        | ✅                                | ❌                       |
| ORB-SLAM3 backend            | ✅                                | ❌                       |
| NVIDIA cuVSLAM backend       | ❌                                | ✅                       |
| Webcam input                 | Possible                         | ✅                       |
| Real-time trajectory         | ✅                                | ✅                       |
| 3D map visualization         | ✅                                | ✅                       |
| Interactive Open3D viewer    | Not the focus                    | ✅                       |
| Persistent visualization map | ✅                                | ✅                       |
| Map save/load                | Supported in ORB-SLAM3 ecosystem | ✅ Custom implementation |

This project should therefore be described as **ORB-SLAM3-style Visual SLAM**, rather than claiming to be an implementation of ORB-SLAM3 itself.

---

# 🎯 Learning Objectives

This project is intended to provide practical experience with:

* Visual SLAM
* Monocular camera tracking
* Camera pose estimation
* 3D landmark visualization
* Coordinate transformations
* Quaternion-to-rotation conversion
* Trajectory estimation
* Visual feature tracking
* GPU-accelerated computer vision
* OpenCV visualization
* Open3D visualization
* Map persistence
* Real-time robotics perception

---

# 🔬 Future Improvements

Potential extensions include:

* [ ] Proper camera calibration
* [ ] Stereo camera support
* [ ] RGB-D camera support
* [ ] Loop-closure integration
* [ ] Improved map optimization
* [ ] Better landmark filtering
* [ ] Real camera calibration parameters
* [ ] ROS 2 integration
* [ ] RViz2 visualization
* [ ] Robot-mounted camera
* [ ] Isaac Sim integration
* [ ] Autonomous robot navigation
* [ ] SLAM + Nav2 integration
* [ ] Pose graph visualization

---

# 🤖 ROS 2 Integration — Future Direction

A natural next step is connecting the SLAM system to ROS 2:

```text
Camera
  │
  ▼
Visual SLAM
  │
  ├── Camera Pose
  ├── Map
  └── Odometry
       │
       ▼
     ROS 2
       │
       ├── TF2
       ├── RViz2
       ├── Nav2
       └── Autonomous Navigation
```

This would allow the visual SLAM system to become part of a complete autonomous robotics pipeline.

---

# 📁 Output Files

A typical SLAM session can generate:

```text
slam_frame_<timestamp>.png
slam_data_<timestamp>.pkl
slam_data_<timestamp>.ply
```

An automatic map save can also be generated when exiting after sufficient trajectory data:

```text
slam_data_autosave.pkl
```

The cleanup routine automatically saves the map when the trajectory contains more than 100 points.

---

# 👨‍💻 Author

**Arun M.**

B.Tech Robotics & Automation

### Interests

* Robotics
* Computer Vision
* Visual SLAM
* ROS 2
* Autonomous Navigation
* Physical AI
* Robot Perception
* NVIDIA Robotics Technologies

---

# ⭐ Project Status

🚧 **Active Learning / Development Project**

This project is part of my hands-on exploration of **Visual SLAM, GPU-accelerated perception, and robotics software development**.

The current implementation focuses on real-time monocular tracking, trajectory visualization, persistent 3D mapping, and interactive visualization using NVIDIA cuVSLAM and Open3D.
