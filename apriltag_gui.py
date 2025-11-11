#!/usr/bin/env python3
"""
apriltag_predictive_timeout_multitag_ll_hardened_mono_optimized_v2.py

PC test: monochrome hot path (grayscale pixels), colored overlays, improved smoothing,
robust prediction, adaptive-rate detection, NT + UDP publisher. Keeps GUI for PC testing.
"""

import time
import threading
import json
import os
import socket
import cv2
import numpy as np
from collections import deque
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Prefer native apriltag; fallback to OpenCV ArUco AprilTag
_HAS_APRILTAG = False
try:
    import apriltag as atag
    _HAS_APRILTAG = True
except Exception:
    _HAS_APRILTAG = False

# ---------------------- Config ----------------------
MAX_DISPLAY_W, MAX_DISPLAY_H = 1200, 800
DEFAULT_TAG_SIZE_M = 0.1524
CAM_IDX = 0
CAPTURE_FPS = 60            # camera target (set lower if hardware can't support)
FRAME_UI_MS = 12

# Performance tuning (start targets; adaptive system will adjust)
DETECTION_RATE_HZ = 60      # initial target detection rate
GUI_RATE_HZ = 20          # UI update rate
MIN_DET_RATE = 8
MAX_DET_RATE = 120

# NetworkTables / UDP
NT_SERVER = "10.xx.yy.2"
UDP_FALLBACK = True
UDP_TARGET = ("10.xx.yy.11", 5800)

# Detection / gating
DEFAULT_DECIMATE = 1
DEFAULT_CONF = 0.18
REPROJ_ERR_THRESH = 2.2

# Motion/adaptive decimation
BLUR_HIGH = 30.0
BLUR_MED = 80.0
ADAPT_DECIMATE_HIGH = 3
ADAPT_DECIMATE_MED = 2
ADAPT_DECIMATE_LOW = 1

# Per-tag timeouts
KEEPALIVE_TIMEOUT = 0.4
REMOVAL_TIMEOUT = 0.8
SCENE_PURGE_TIMEOUT = 1.2

# Dynamics
VELOCITY_DECAY = 0.82
MAX_PREDICT_DISTANCE = 0.45
MIN_SCALE_PX = 6.0
MAX_SCALE_PX = 5000.0

# Optical flow (PyrLK)
_LK_PARAMS = dict(winSize=(21,21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# Kalman defaults
TRACK_Q = 0.02
TRACK_R = 4.0

# EMA smoothing defaults
EMA_ALPHA_POS = 0.28
EMA_ALPHA_POSE = 0.22

# Pose median smoothing window (set 0 to disable)
POSE_MEDIAN_WINDOW = 5

# Safety/cpu thresholds
PROCESS_TIME_HIGH_MS = 18.0   # if average process loop > this, reduce DETECTION_RATE_HZ
PROCESS_TIME_LOW_MS = 8.0     # if average < this, try to increase DETECTION_RATE_HZ

# Fonts
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    from PIL import ImageFont as _IF
    FONT = _IF.load_default()

# -------------------- Utilities --------------------
def default_camera_matrix(shape):
    h, w = shape
    f = 0.9 * max(w, h)
    K = np.array([[f,0,w/2.0],[0,f,h/2.0],[0,0,1]], dtype=np.float64)
    D = np.zeros((5,1), dtype=np.float64)
    return K, D

def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def choose_decimate(base, blurv):
    if blurv < BLUR_HIGH:
        return max(base, ADAPT_DECIMATE_HIGH)
    elif blurv < BLUR_MED:
        return max(base, ADAPT_DECIMATE_MED)
    else:
        return max(base, ADAPT_DECIMATE_LOW)

def clamp(v, a, b):
    return max(a, min(b, v))

def polygon_area(pts):
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

# -------------------- Detector wrapper --------------------
cv2.setUseOptimized(True)
USE_OPENCV_ARUCO = False
DICT_OBJ = None
_ARUCO_PARAMS = None
if not _HAS_APRILTAG:
    if hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        USE_OPENCV_ARUCO = True
        DICT_OBJ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        try:
            _ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
            _ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            _ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    else:
        raise RuntimeError("Install 'apriltag' or 'opencv-contrib-python' for AprilTag support.")

class Detector:
    def __init__(self):
        self.use_native = _HAS_APRILTAG
        if self.use_native:
            self.det = atag.Detector()
        else:
            self.dict = DICT_OBJ
            self.params = _ARUCO_PARAMS
            try:
                if hasattr(cv2.aruco, "ArucoDetector"):
                    self.opencv_detector = cv2.aruco.ArucoDetector(self.dict, self.params)
                else:
                    self.opencv_detector = None
            except Exception:
                self.opencv_detector = None

    def detect(self, gray):
        out = []
        if self.use_native:
            dets = self.det.detect(gray)
            for d in dets:
                corners = [(float(p[0]), float(p[1])) for p in d.corners]
                out.append({"id": int(d.tag_id), "corners": corners})
        else:
            if self.opencv_detector is not None:
                corners, ids, _ = self.opencv_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)
            if ids is None:
                return []
            ids = np.array(ids).reshape(-1)
            for i, c in enumerate(corners):
                pts = np.array(c).reshape(-1, 2)
                corners_list = [(float(x), float(y)) for x, y in pts]
                out.append({"id": int(ids[i]), "corners": corners_list})
        return out

# -------------------- Pose --------------------
def solve_pose(corners, tag_m, K, D):
    obj = np.array([
        [-tag_m/2, -tag_m/2, 0.0],
        [ tag_m/2, -tag_m/2, 0.0],
        [ tag_m/2,  tag_m/2, 0.0],
        [-tag_m/2,  tag_m/2, 0.0]
    ], dtype=np.float32)
    imgp = np.array(corners, dtype=np.float32).reshape(4, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            return None
    if rvec is None or tvec is None:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    err = np.mean(np.linalg.norm(proj - imgp, axis=1))
    if not np.isfinite(err) or err > REPROJ_ERR_THRESH:
        return None
    return {"rvec": rvec, "tvec": tvec, "err": err}

# -------------------- Smoothers --------------------
class EMASmoother:
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.v = None
    def update(self, val):
        v = np.array(val, dtype=np.float64).reshape(-1)
        if self.v is None:
            self.v = v
        else:
            self.v = self.alpha * v + (1 - self.alpha) * self.v
        return self.v.copy()

class MedianBuffer:
    def __init__(self, window):
        self.window = int(max(1, window))
        self.q = deque(maxlen=self.window)
    def push(self, val):
        self.q.append(np.array(val, dtype=np.float64).reshape(-1))
    def median(self):
        if not self.q:
            return None
        arr = np.stack(self.q, axis=0)
        return np.median(arr, axis=0)

# -------------------- Kalman tracker --------------------
class BoxTracker:
    def __init__(self, q=TRACK_Q, r=TRACK_R):
        self.q = float(q); self.r = float(r)
        self.x = None
        self.P = None
        self.F = np.eye(6)
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6) * self.q
        self.R = np.eye(3) * self.r
        self.last_ts = time.time()
        self.unseen_count = 0.0
    def init(self, cx, cy, s):
        self.x = np.array([cx, cy, s, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6) * 16.0
        self.last_ts = time.time()
        self.unseen_count = 0.0
    def predict(self):
        if self.x is None: return None
        now = time.time(); dt = max(1e-3, now - self.last_ts); self.last_ts = now
        F = np.eye(6); F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        self.F = F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.x[3] *= VELOCITY_DECAY
        self.x[4] *= VELOCITY_DECAY
        self.x[5] *= VELOCITY_DECAY
        self.unseen_count += dt
        return self.x.copy()
    def update(self, cx, cy, s):
        if self.x is None:
            self.init(cx, cy, s)
            return self.x.copy()
        z = np.array([cx, cy, s], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6); self.P = (I - K @ self.H) @ self.P
        self.unseen_count = 0.0
        return self.x.copy()
    def get(self):
        if self.x is None: return None
        return float(self.x[0]), float(self.x[1]), float(self.x[2])
    def seconds_unseen(self):
        return float(self.unseen_count)

# -------------------- Optical flow helpers --------------------
def corners_to_np(corners):
    return np.array(corners, dtype=np.float32).reshape(-1, 1, 2)

def np_to_corners(pts):
    pts = np.array(pts).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]

# -------------------- Limelight values --------------------
def compute_ll_values(corners, K, w, h):
    pts = np.array(corners, dtype=np.float64).reshape(4,2)
    cx = np.mean(pts[:,0]); cy = np.mean(pts[:,1])
    fx = float(K[0,0]); fy = float(K[1,1]); cx0 = float(K[0,2]); cy0 = float(K[1,2])
    tx_rad = np.arctan2(cx - cx0, fx)
    ty_rad = np.arctan2(cy0 - cy, fy)
    tx_deg = float(np.degrees(tx_rad))
    ty_deg = float(np.degrees(ty_rad))
    area_px = polygon_area(pts)
    ta_percent = float(100.0 * area_px / (w * h))
    return tx_deg, ty_deg, ta_percent

# -------------------- IO / Buffers --------------------
class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0
    def put(self, frame):
        with self.lock:
            self.frame = frame
            self.ts = time.time()
    def get(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.ts)

# -------------------- NT Publisher --------------------
class NTPublisher(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.sock = None
        if UDP_FALLBACK:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setblocking(False)
            except Exception:
                self.sock = None
        self.nt = None
    def run(self):
        try:
            from networktables import NetworkTables
            NetworkTables.initialize(server=NT_SERVER)
            self.nt = NetworkTables.getTable("Vision")
        except Exception:
            self.nt = None
        while self.app.process_running:
            payload = None
            with self.app.nt_lock:
                if self.app.nt_queue:
                    payload = self.app.nt_queue.pop(0)
            if payload:
                try:
                    if self.nt is not None and self.app.nt_var.get():
                        ts = payload.get("timestamp", time.time())
                        self.nt.putNumber("timestamp", ts)
                        self.nt.putNumber("tag_count", len(payload.get("tags", [])))
                        self.nt.putString("tagIDs", ",".join(str(t["id"]) for t in payload.get("tags",[])))
                        for rec in payload.get("tags", []):
                            pid = rec["id"]
                            pref = f"{pid}_"
                            self.nt.putNumber(pref+"tx", rec["tx"])
                            self.nt.putNumber(pref+"ty", rec["ty"])
                            self.nt.putNumber(pref+"ta", rec["ta"])
                            self.nt.putNumber(pref+"tx_m", rec["tvec"][0])
                            self.nt.putNumber(pref+"ty_m", rec["tvec"][1])
                            self.nt.putNumber(pref+"tz_m", rec["tvec"][2])
                            self.nt.putNumber(pref+"reproj", rec.get("reproj_err", 0.0))
                    if UDP_FALLBACK and self.sock:
                        try:
                            self.sock.sendto(json.dumps(payload).encode("utf8"), UDP_TARGET)
                        except Exception:
                            pass
                except Exception:
                    pass
            time.sleep(0.004)

# -------------------- App --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("AprilTag Mono Feed Optimized v2")
        ctrl = tk.Frame(root); ctrl.pack(fill="x", padx=6, pady=6)

        tk.Button(ctrl, text="Open Images", command=self.open_images).pack(side="left", padx=3)
        tk.Button(ctrl, text="Load Intrinsics", command=self.load_intrinsics).pack(side="left", padx=3)
        tk.Button(ctrl, text="Tag Size (m)", command=self.set_tag_size).pack(side="left", padx=3)

        self.clahe_var = tk.IntVar(value=1); tk.Checkbutton(ctrl, text="CLAHE", variable=self.clahe_var).pack(side="left", padx=4)
        tk.Label(ctrl, text="Gamma").pack(side="left"); self.gamma_e = tk.Entry(ctrl, width=4); self.gamma_e.insert(0,"1.25"); self.gamma_e.pack(side="left")
        tk.Label(ctrl, text="Base decimate").pack(side="left"); self.dec_e = tk.Entry(ctrl, width=3); self.dec_e.insert(0,str(DEFAULT_DECIMATE)); self.dec_e.pack(side="left")
        tk.Label(ctrl, text="EMA pos").pack(side="left"); self.ema_pos_e = tk.Entry(ctrl, width=4); self.ema_pos_e.insert(0,str(EMA_ALPHA_POS)); self.ema_pos_e.pack(side="left")
        tk.Label(ctrl, text="EMA pose").pack(side="left"); self.ema_pose_e = tk.Entry(ctrl, width=4); self.ema_pose_e.insert(0,str(EMA_ALPHA_POSE)); self.ema_pose_e.pack(side="left")

        tk.Button(ctrl, text="Camera Mode", command=self.toggle_camera).pack(side="left", padx=6)
        self.nt_var = tk.IntVar(value=0); tk.Checkbutton(ctrl, text="Publish NT", variable=self.nt_var).pack(side="left", padx=4)

        tk.Label(ctrl, text="Whitelist").pack(side="left", padx=(12,2))
        self.wl_e = tk.Entry(ctrl, width=12); self.wl_e.pack(side="left")
        tk.Label(ctrl, text="Blacklist").pack(side="left", padx=(6,2))
        self.bl_e = tk.Entry(ctrl, width=12); self.bl_e.pack(side="left")
        tk.Button(ctrl, text="Apply Filters", command=self.apply_filters).pack(side="left", padx=4)

        self.info_label = tk.Label(ctrl, text="Ready"); self.info_label.pack(side="left", padx=8)
        self.canvas = tk.Canvas(root, bg="black"); self.canvas.pack(fill="both", expand=True, padx=6, pady=6)

        # state
        self.files = []; self.index = 0; self.cache = {}
        self.tag_size_m = DEFAULT_TAG_SIZE_M
        self.K = None; self.D = None
        self.detector = Detector()

        # trackers and smoothing
        self.trackers = {}
        self.pos_smoothers = {}   # id -> EMASmoother (for tvec)
        self.pose_smoothers = {}  # id -> EMASmoother (for rvec)
        self.pose_medians = {}    # id -> MedianBuffer
        self.last_corners = {}
        self.lk_last_pts = {}
        self.lk_prev_gray = None

        # watchdogs
        self.scene_unseen_start = None

        # frames & threads
        self.frame_buf = LatestFrameBuffer()
        self.result_buf = LatestFrameBuffer()
        self.capture_running = False
        self.process_running = False
        self.tkimg = None

        # perf & NT queue
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
        self.corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.lut_gamma = None
        self._lut_gamma_val = None
        self.gray_buf = None
        self.last_gui_time = 0.0
        self.nt_queue = []
        self.nt_lock = threading.Lock()
        self.nt_pub = None

        # filters
        self.whitelist = None
        self.blacklist = None

        # adaptive control
        self.det_rate = float(DETECTION_RATE_HZ)
        self.process_time_hist = deque(maxlen=30)
        self.detect_count = 0
        self.gui_count = 0
        self.fps_last_report = time.time()

    def open_images(self):
        files = filedialog.askopenfilenames(title="Select images", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not files: return
        self.files = list(files); self.index = 0; self.cache.clear()
        self.info_label.config(text=f"Loaded {len(self.files)} images"); self.show_current()

    def set_tag_size(self):
        val = simpledialog.askfloat("Tag Size (m)", "Enter tag side length (meters)", initialvalue=self.tag_size_m, minvalue=0.001)
        if val: self.tag_size_m = float(val)

    def load_intrinsics(self):
        path = filedialog.askopenfilename(title="Load intrinsics JSON", filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path,"r") as fh:
                j = json.load(fh)
            fx = float(j["fx"]); fy = float(j.get("fy", j["fx"]))
            cx = float(j["cx"]); cy = float(j["cy"])
            self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            d = j.get("dist", [0,0,0,0,0]); self.D = np.array(d, dtype=np.float64).reshape(-1,1)
            messagebox.showinfo("Intrinsics", f"Loaded {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error loading intrinsics", str(e))

    def apply_filters(self):
        self.whitelist = parse_id_list(self.wl_e.get()) if self.wl_e.get().strip() else None
        self.blacklist = parse_id_list(self.bl_e.get()) if self.bl_e.get().strip() else None
        wl = f"WL={sorted(self.whitelist) if self.whitelist else 'all'}"; bl = f"BL={sorted(self.blacklist) if self.blacklist else 'none'}"
        self.info_label.config(text=f"Filters applied: {wl}, {bl}")

    def show_current(self):
        if not self.files: return
        f = self.files[self.index]
        if f in self.cache:
            pil = self.cache[f]
        else:
            pil = self.process_and_draw(f)
            self.cache[f] = pil
        disp = self.resize_for_display(pil.copy())
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.delete("all"); self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def resize_for_display(self, pil):
        w, h = pil.size
        if w <= MAX_DISPLAY_W and h <= MAX_DISPLAY_H: return pil
        scale = min(MAX_DISPLAY_W/w, MAX_DISPLAY_H/h)
        return pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # ---------- Core processing ----------
    def process_and_draw(self, frame_or_path):
        t0 = time.time()
        # load frame
        if isinstance(frame_or_path, str):
            arr = np.fromfile(frame_or_path, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return Image.new("RGB",(640,480),(0,0,0))
        else:
            bgr = frame_or_path
        h, w = bgr.shape[:2]
        if self.K is None:
            self.K, self.D = default_camera_matrix((h, w))

        # controls & dynamic values
        try: base_dec = max(1, int(self.dec_e.get()))
        except: base_dec = DEFAULT_DECIMATE
        try: ema_pos_alpha = float(self.ema_pos_e.get())
        except: ema_pos_alpha = EMA_ALPHA_POS
        try: ema_pose_alpha = float(self.ema_pose_e.get())
        except: ema_pose_alpha = EMA_ALPHA_POSE

        # convert to grayscale once
        if self.gray_buf is None or self.gray_buf.shape != (h,w):
            self.gray_buf = np.empty((h,w), dtype=np.uint8)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY, dst=self.gray_buf)

        # preproc (CLAHE + gamma)
        if self.clahe_var.get():
            gray_proc = self.clahe.apply(gray_full)
            try:
                gamma = float(self.gamma_e.get() or 1.25)
            except:
                gamma = 1.25
            if abs(gamma - 1.0) > 1e-3:
                if self.lut_gamma is None or self._lut_gamma_val != gamma:
                    inv = 1.0 / gamma
                    self.lut_gamma = np.array([((i/255.0)**inv)*255 for i in range(256)], dtype=np.uint8)
                    self._lut_gamma_val = gamma
                gray_proc = cv2.LUT(gray_proc, self.lut_gamma)
        else:
            gray_proc = gray_full

        # detection on grayscale (monochrome). adaptive decimation by blur
        blurv = laplacian_var(gray_proc)
        dec = choose_decimate(base_dec, blurv)

        if dec > 1:
            small = cv2.resize(gray_proc, (max(1, w//dec), max(1, h//dec)), interpolation=cv2.INTER_LINEAR)
            dets = self.detector.detect(small)
            for d in dets:
                d["corners"] = [(x*dec, y*dec) for (x, y) in d["corners"]]
        else:
            dets = self.detector.detect(gray_proc)

        # filter IDs
        if self.whitelist is not None:
            dets = [d for d in dets if d["id"] in self.whitelist]
        if self.blacklist is not None:
            dets = [d for d in dets if d["id"] not in self.blacklist]

        now = time.time()
        visible_ids = set()
        ll_table = []
        valid_xyz = []

        # Prepare visual base: grayscale->BGR so feed is mono but overlays retain color
        vis_bgr = cv2.cvtColor(gray_proc, cv2.COLOR_GRAY2BGR)

        # Process detections
        for d in dets:
            pid = d["id"]; corners = np.array(d["corners"], dtype=np.float32)
            if corners.shape != (4, 2): continue

            # refined corners only when robust area
            area_px = polygon_area(corners)
            if area_px > 30.0:
                try:
                    cv2.cornerSubPix(gray_proc, corners, winSize=(5,5), zeroZone=(-1,-1),
                                     criteria=self.corner_subpix_criteria)
                except Exception:
                    pass

            d["corners"] = [(float(x), float(y)) for x, y in corners.reshape(4,2)]
            visible_ids.add(pid)

            # draw detection polygon (green)
            pts = np.array(d["corners"], dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(vis_bgr, [pts], True, (0,220,0), 2, cv2.LINE_AA)

            # tracker update
            diag = float(np.linalg.norm(corners[0] - corners[2]))
            cx = float(np.mean(corners[:,0])); cy = float(np.mean(corners[:,1]))
            tr = self.trackers.get(pid)
            if tr is None:
                tr = BoxTracker(q=TRACK_Q, r=TRACK_R); tr.init(cx, cy, max(MIN_SCALE_PX, diag)); self.trackers[pid] = tr
            else:
                tr.update(cx, cy, max(MIN_SCALE_PX, diag))
            self.last_corners[pid] = corners.copy()
            self.lk_last_pts[pid] = corners_to_np(corners)

            # pose: only when reasonably large to avoid noisy solves
            pose = None
            if diag >= 14.0:
                pose = solve_pose(d["corners"], self.tag_size_m, self.K, self.D)
            tx_deg, ty_deg, ta_pct = compute_ll_values(d["corners"], self.K, w, h)

            if pose:
                tvec = pose["tvec"].reshape(3); rvec = pose["rvec"].reshape(3)
                # position smoothing
                sm_p = self.pos_smoothers.get(pid)
                if sm_p is None or abs(sm_p.alpha - ema_pos_alpha) > 1e-9:
                    sm_p = EMASmoother(alpha=ema_pos_alpha); self.pos_smoothers[pid] = sm_p
                t_sm = sm_p.update(tvec)

                # pose smoothing and median buffer
                sm_r = self.pose_smoothers.get(pid)
                if sm_r is None or abs(sm_r.alpha - ema_pose_alpha) > 1e-9:
                    sm_r = EMASmoother(alpha=ema_pose_alpha); self.pose_smoothers[pid] = sm_r
                r_sm = sm_r.update(rvec)

                if POSE_MEDIAN_WINDOW > 1:
                    mb = self.pose_medians.get(pid)
                    if mb is None:
                        mb = MedianBuffer(POSE_MEDIAN_WINDOW); self.pose_medians[pid] = mb
                    mb.push(np.hstack([t_sm, r_sm]))
                    med = mb.median()
                    if med is not None:
                        t_out = med[:3]; r_out = med[3:6]
                    else:
                        t_out = t_sm; r_out = r_sm
                else:
                    t_out = t_sm; r_out = r_sm

                valid_xyz.append((pid, t_out.copy(), 1.0))
                ll_table.append({
                    "id": int(pid),
                    "tx": float(tx_deg),
                    "ty": float(ty_deg),
                    "ta": float(ta_pct),
                    "tvec": [float(t_out[0]), float(t_out[1]), float(t_out[2])],
                    "rvec": [float(r_out[0]), float(r_out[1]), float(r_out[2])],
                    "reproj_err": float(pose["err"])
                })

                # overlay text near centroid (keeps overlay colored)
                cx_i, cy_i = int(round(cx)), int(round(cy))
                txt = f"ID{pid} tx={tx_deg:.1f}° ty={ty_deg:.1f}° ta={ta_pct:.2f}% Z={t_out[2]:.3f}m"
                im_tmp = Image.new("RGB", (1,1)); draw_tmp = ImageDraw.Draw(im_tmp)
                bbox = draw_tmp.textbbox((0,0), txt, font=FONT)
                tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
                x0, y0 = clamp(cx_i - tw//2, 2, w-2-tw), clamp(cy_i - th - 8, 2, h-2-th)
                cv2.rectangle(vis_bgr, (int(x0)-4, int(y0)-4), (int(x0)+tw+4, int(y0)+th+6), (0,0,0), -1)
                cv2.putText(vis_bgr, txt, (int(x0), int(y0+th)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)

        # Scene watchdog and purge when all invisible for extended time
        if len(visible_ids) > 0:
            self.scene_unseen_start = None
        else:
            if self.scene_unseen_start is None:
                self.scene_unseen_start = now
            else:
                if now - self.scene_unseen_start >= SCENE_PURGE_TIMEOUT:
                    self.trackers.clear()
                    self.pos_smoothers.clear()
                    self.pose_smoothers.clear()
                    self.pose_medians.clear()
                    self.last_corners.clear()
                    self.lk_last_pts.clear()
                    self.scene_unseen_start = None

        # Prediction for trackers not currently visible: LK first, then Kalman
        if self.lk_prev_gray is None:
            self.lk_prev_gray = gray_full.copy()

        for pid, tr in list(self.trackers.items()):
            if pid in visible_ids:
                continue

            seconds_unseen = tr.seconds_unseen()
            if seconds_unseen > REMOVAL_TIMEOUT:
                self.trackers.pop(pid, None)
                self.pos_smoothers.pop(pid, None)
                self.pose_smoothers.pop(pid, None)
                self.pose_medians.pop(pid, None)
                self.last_corners.pop(pid, None)
                self.lk_last_pts.pop(pid, None)
                continue

            if seconds_unseen > KEEPALIVE_TIMEOUT:
                # slowly decay velocities, still call predict for internal damping
                tr.predict()
                continue

            p0 = self.lk_last_pts.get(pid)
            drew = False
            if p0 is not None and p0.size != 0:
                try:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.lk_prev_gray, gray_full, p0, None, **_LK_PARAMS)
                    if p1 is not None:
                        st = st.reshape(-1)
                        good_cnt = int(np.count_nonzero(st==1))
                        if good_cnt >= 3:
                            pts = p1.reshape(-1,2)
                            if pts.shape[0] >= 4:
                                new_corners = np.array(pts[:4], dtype=np.float32)
                                new_corners[:,0] = np.clip(new_corners[:,0], 1, w-2)
                                new_corners[:,1] = np.clip(new_corners[:,1], 1, h-2)
                                self.last_corners[pid] = new_corners.copy()
                                self.lk_last_pts[pid] = corners_to_np(new_corners)
                                cx = float(np.mean(new_corners[:,0])); cy = float(np.mean(new_corners[:,1]))
                                diag = float(np.linalg.norm(new_corners[0] - new_corners[2]))
                                tr.update(cx, cy, clamp(diag, MIN_SCALE_PX, MAX_SCALE_PX))
                                pts_i = np.array(np_to_corners(new_corners), dtype=np.int32).reshape((-1,1,2))
                                cv2.polylines(vis_bgr, [pts_i], True, (0,180,255), 2, cv2.LINE_AA)
                                drew = True
                except Exception:
                    drew = False

            if not drew:
                pred = tr.predict()
                if pred is None:
                    continue
                cx, cy, s = tr.get()
                diag_img = np.hypot(w, h)
                max_move = MAX_PREDICT_DISTANCE * diag_img
                last_c = self.last_corners.get(pid)
                if last_c is not None:
                    last_cx = float(np.mean(last_c[:,0])); last_cy = float(np.mean(last_c[:,1]))
                    dist = np.hypot(cx - last_cx, cy - last_cy)
                    if dist > max_move:
                        dx = cx - last_cx; dy = cy - last_cy
                        factor = max_move / (dist + 1e-6)
                        cx = last_cx + dx * factor
                        cy = last_cy + dy * factor
                        tr.x[0], tr.x[1] = cx, cy
                        tr.x[3] *= VELOCITY_DECAY; tr.x[4] *= VELOCITY_DECAY
                pad = int(0.5 * min(w, h))
                cx = clamp(cx, -pad, w-1+pad); cy = clamp(cy, -pad, h-1+pad)
                s = clamp(s, MIN_SCALE_PX, MAX_SCALE_PX)
                half = max(6, int(s * 0.5))
                p1 = (int(cx - half), int(cy - half)); p2 = (int(cx + half), int(cy + half))
                cv2.rectangle(vis_bgr, p1, p2, (0,160,255), 2, cv2.LINE_AA)

        self.lk_prev_gray = gray_full.copy()

        # enqueue NT payload
        payload = {"timestamp": time.time(), "tags": ll_table}
        with self.nt_lock:
            self.nt_queue.append(payload)
            if len(self.nt_queue) > 8:
                self.nt_queue = self.nt_queue[-8:]

        # Convert to PIL only for GUI (mono feed preserved because vis_bgr built from gray)
        rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for d in dets:
            pid = d["id"]
            tx = int(d["corners"][0][0]); ty = int(d["corners"][0][1] - 20)
            txt = f"id={pid}"
            bbox = draw.textbbox((0,0), txt, font=FONT)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            draw.rectangle([tx-3, ty-3, tx+tw+3, ty+th+3], fill=(0,0,0))
            draw.text((tx,ty), txt, fill=(255,255,255), font=FONT)
        if not dets and not self.trackers:
            draw.text((8,8), "No AprilTags detected", fill=(255,255,255), font=FONT)

        # profiling & adaptive control
        t1 = time.time()
        proc_ms = (t1 - t0) * 1000.0
        self.process_time_hist.append(proc_ms)
        self.detect_count += 1

        # adjust detection rate every N frames
        if len(self.process_time_hist) == self.process_time_hist.maxlen:
            avg_ms = float(np.mean(self.process_time_hist))
            if avg_ms > PROCESS_TIME_HIGH_MS and self.det_rate > MIN_DET_RATE:
                # back off detection rate
                self.det_rate = max(MIN_DET_RATE, self.det_rate * 0.85)
            elif avg_ms < PROCESS_TIME_LOW_MS and self.det_rate < MAX_DET_RATE:
                self.det_rate = min(MAX_DET_RATE, self.det_rate * 1.08)
            # clear some history to avoid oscillation
            self.process_time_hist.clear()

        return pil

    # ---------- Camera control ----------
    def toggle_camera(self):
        if self.capture_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(int(CAM_IDX), cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(int(CAM_IDX))
            if not self.cap.isOpened():
                messagebox.showerror("Camera", "Cannot open camera"); return
            # pick a reasonable capture size; lower if CPU bound
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                ae = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                if ae != -1:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Camera open", str(e)); return

        self.capture_running = True; self.process_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

        # start NT publisher
        self.nt_pub = NTPublisher(self)
        self.nt_pub.start()

        self.root.after(FRAME_UI_MS, self._ui_loop)
        self.info_label.config(text="Camera running")

    def stop_camera(self):
        self.capture_running = False; self.process_running = False
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.info_label.config(text="Camera stopped")

    def _capture_loop(self):
        while self.capture_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005); continue
            self.frame_buf.put(frame)
            time.sleep(0.001)

    def _process_loop(self):
        last = 0.0
        while self.process_running:
            # adapt detection timestep from self.det_rate
            min_dt = 1.0 / max(1.0, self.det_rate)
            tnow = time.time()
            if tnow - last < min_dt:
                time.sleep(max(0.0, min_dt - (tnow-last)))
            last = time.time()
            frame, ts = self.frame_buf.get()
            if frame is None:
                time.sleep(0.002); continue
            pil = self.process_and_draw(frame)
            # GUI controlled separately
            if time.time() - self.last_gui_time > (1.0 / max(1, GUI_RATE_HZ)):
                self.last_gui_time = time.time()
                disp = self.resize_for_display(pil.copy())
                self.result_buf.put(disp)
            time.sleep(0.001)

    def _ui_loop(self):
        disp, ts = self.result_buf.get()
        if disp is not None:
            try:
                self.tkimg = ImageTk.PhotoImage(disp)
                self.canvas.delete("all")
                self.canvas.create_image(0,0,anchor="nw", image=self.tkimg)
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
                # show some perf info
                now = time.time()
                if now - self.fps_last_report > 1.0:
                    self.info_label.config(text=f"det_rate={self.det_rate:.1f}Hz  GUI={GUI_RATE_HZ}Hz")
                    self.detect_count = 0
                    self.gui_count = 0
                    self.fps_last_report = now
            except Exception:
                pass
        if self.capture_running:
            self.root.after(FRAME_UI_MS, self._ui_loop)

# -------------------- Helpers --------------------
def parse_id_list(text):
    s = text.strip()
    if not s: return None
    out = set()
    for p in s.split(","):
        p = p.strip()
        if not p: continue
        if "-" in p:
            try:
                a, b = p.split("-",1); a = int(a); b = int(b)
                lo, hi = (a, b) if a <= b else (b, a)
                out.update(range(lo, hi+1))
            except:
                pass
        else:
            try: out.add(int(p))
            except: pass
    return out if out else None

# -------------------- Run --------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("1100x720")
    root.mainloop()

if __name__ == "__main__":
    main()
