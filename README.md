# Python-April-Vision

A robust, **monochrome AprilTag detection and tracking system** with predictive timeouts, Kalman filtering, optical flow, adaptive-rate detection, and NetworkTables/UDP publishing.  
Designed for **FRC/FTC robotics** but also useful for PC testing and simulation.

This is the original Python version of [April-Vision](https://github.com/TomAs-1226/April-vision).

---

## ✨ Features
- ✅ Native [`apriltag`](https://pypi.org/project/apriltag/) support, with fallback to OpenCV ArUco AprilTag.
- 🎥 Camera capture or image file testing with Tkinter GUI.
- 🖤 Monochrome hot path (grayscale feed) with **colored overlays** for clarity.
- 🔄 Adaptive decimation based on blur (fast on modest hardware).
- 📦 Kalman + optical flow prediction for tag persistence when occluded.
- 📡 Publishes to **NetworkTables** (FRC standard) and UDP fallback.
- ⚙️ Configurable smoothing (EMA + median), reprojection error gating, and adaptive detection rate.

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/TomAs-1226/Python-April-vision.git
cd Python-April-vision
