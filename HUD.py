#!/usr/bin/env python3
import argparse
import time
import math
from collections import deque
 
import cv2
import numpy as np
 
# -----------------------------
# Small helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))
 
def iou(a, b):
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)
 
def point_in_poly(pt, poly):
    # poly: Nx2 int
    return cv2.pointPolygonTest(poly.astype(np.int32), (float(pt[0]), float(pt[1])), False) >= 0
 
def ema(prev, cur, alpha):
    if prev is None:
        return cur
    return (alpha * prev) + ((1.0 - alpha) * cur)
 
# -----------------------------
# Simple tracker to keep boxes briefly
# -----------------------------
class SimpleTracker:
    def __init__(self, iou_thresh=0.3, ttl=8):
        self.iou_thresh = iou_thresh
        self.ttl = ttl
        self.next_id = 1
        self.tracks = {}  # id -> dict(box=..., cls=..., conf=..., ttl=...)
 
    def update(self, dets):
        """
        dets: list of dict: {box:(x1,y1,x2,y2), cls:str, conf:float}
        """
        # decrement ttl
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["ttl"] -= 1
            if self.tracks[tid]["ttl"] <= 0:
                del self.tracks[tid]
 
        used = set()
        # match detections to existing tracks by IoU (same class)
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, d in enumerate(dets):
                if j in used:
                    continue
                if d["cls"] != tr["cls"]:
                    continue
                val = iou(tr["box"], d["box"])
                if val > best_iou:
                    best_iou = val
                    best_j = j
 
            if best_j != -1 and best_iou >= self.iou_thresh:
                d = dets[best_j]
                used.add(best_j)
                self.tracks[tid] = {
                    "box": d["box"],
                    "cls": d["cls"],
                    "conf": d["conf"],
                    "ttl": self.ttl,
                }
 
        # create new tracks for unmatched dets
        for j, d in enumerate(dets):
            if j in used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": d["box"],
                "cls": d["cls"],
                "conf": d["conf"],
                "ttl": self.ttl,
            }
 
        # return list for drawing
        out = []
        for tid, tr in self.tracks.items():
            out.append({"id": tid, **tr})
        return out
 
# -----------------------------
# Lane detection (Canny + Hough) + smoothing
# -----------------------------
def detect_lane_lines(frame_bgr, roi_poly, canny1, canny2, hough_thresh, min_len, max_gap):
    h, w = frame_bgr.shape[:2]
 
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
 
    edges = cv2.Canny(blur, canny1, canny2)
 
    # Mask ROI
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi_poly.astype(np.int32)], 255)
    masked = cv2.bitwise_and(edges, mask)
 
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
 
    left = []
    right = []
 
    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.4:  # ignore near-horizontal
                continue
            if slope < 0:
                left.append((x1, y1, x2, y2))
            else:
                right.append((x1, y1, x2, y2))
 
    def fit_line(segments):
        if len(segments) == 0:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in segments:
            xs += [x1, x2]
            ys += [y1, y2]
        # Fit x = m*y + b (more stable for near-vertical)
        y = np.array(ys, dtype=np.float32)
        x = np.array(xs, dtype=np.float32)
        m, b = np.polyfit(y, x, 1)
        return m, b
 
    left_fit = fit_line(left)
    right_fit = fit_line(right)
 
    # Convert to endpoints inside ROI (use top_y and bottom_y from ROI)
    y_top = int(np.min(roi_poly[:, 1]))
    y_bot = int(np.max(roi_poly[:, 1]))
 
    def endpoints(fit):
        if fit is None:
            return None
        m, b = fit
        x_top = int(m * y_top + b)
        x_bot = int(m * y_bot + b)
        return (x_top, y_top, x_bot, y_bot)
 
    return endpoints(left_fit), endpoints(right_fit), masked
 
# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0, help="camera index (0) or use --video for file")
    ap.add_argument("--video", type=str, default=None, help="path to video file")
    ap.add_argument("--prototxt", type=str, default="MobileNetSSD_deploy.prototxt")
    ap.add_argument("--model", type=str, default="MobileNetSSD_deploy.caffemodel")
    ap.add_argument("--conf", type=float, default=0.40, help="min confidence")
    ap.add_argument("--skip", type=int, default=0, help="skip N frames each loop (performance)")
    args = ap.parse_args()
 
    # Video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture()
        cap.open(args.source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("CAMERA RESOLUTION",
             cap.get(cv2.CAP_PROP_FRAME_WIDTH),
             cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return
 
    # Load DNN
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
 
    # Controls window (separate!)
    CTRL_WIN = "HUD Controls"
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 520, 420)
 
    # --- Sliders (what they do is explained below) ---
    # Trapezoid geometry
    cv2.createTrackbar("Zone Center X %", CTRL_WIN, 50, 100, lambda x: None)   # center horizontally
    cv2.createTrackbar("Zone Top Y %",   CTRL_WIN, 55, 95,  lambda x: None)   # where trapezoid starts
    cv2.createTrackbar("Zone Bottom Y %",CTRL_WIN, 92, 99,  lambda x: None)   # near bottom
    cv2.createTrackbar("Zone Top W %",   CTRL_WIN, 22, 90,  lambda x: None)   # width at top
    cv2.createTrackbar("Zone Bot W %",   CTRL_WIN, 92, 100, lambda x: None)   # width at bottom
 
    # Lane parameters
    cv2.createTrackbar("Canny1",         CTRL_WIN, 60, 255, lambda x: None)
    cv2.createTrackbar("Canny2",         CTRL_WIN, 150, 255, lambda x: None)
    cv2.createTrackbar("Hough Thresh",   CTRL_WIN, 40, 200, lambda x: None)
    cv2.createTrackbar("Min Line Len",   CTRL_WIN, 50, 400, lambda x: None)
    cv2.createTrackbar("Max Line Gap",   CTRL_WIN, 90, 400, lambda x: None)
    cv2.createTrackbar("Smooth %",       CTRL_WIN, 70, 99,  lambda x: None)   # higher = smoother
 
    # Object filtering / tracking
    cv2.createTrackbar("Filter in Zone", CTRL_WIN, 1, 1,    lambda x: None)   # 1=only show cars in zone
    cv2.createTrackbar("Keep Boxes TTL", CTRL_WIN, 8, 30,   lambda x: None)   # keep detection a few frames
 
    # State for smoothing
    prev_left = None
    prev_right = None
 
    # Tracker
    tracker = SimpleTracker(iou_thresh=0.25, ttl=8)
 
    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0
 
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
   
    while True:
        # optional frame skipping for speed
        for _ in range(args.skip):
            cap.grab()
 
        ok, frame = cap.read()
        frame = cv2.resize(frame, (640,360)) 
        if not ok:
            break
 
        h, w = frame.shape[:2]
 
        # --- Read sliders ---
        cx_pct = cv2.getTrackbarPos("Zone Center X %", CTRL_WIN) / 100.0
        topy_pct = cv2.getTrackbarPos("Zone Top Y %", CTRL_WIN) / 100.0
        boty_pct = cv2.getTrackbarPos("Zone Bottom Y %", CTRL_WIN) / 100.0
        topw_pct = cv2.getTrackbarPos("Zone Top W %", CTRL_WIN) / 100.0
        botw_pct = cv2.getTrackbarPos("Zone Bot W %", CTRL_WIN) / 100.0
 
        canny1 = cv2.getTrackbarPos("Canny1", CTRL_WIN)
        canny2 = cv2.getTrackbarPos("Canny2", CTRL_WIN)
        hth   = cv2.getTrackbarPos("Hough Thresh", CTRL_WIN)
        minL  = cv2.getTrackbarPos("Min Line Len", CTRL_WIN)
        maxG  = cv2.getTrackbarPos("Max Line Gap", CTRL_WIN)
        smooth_pct = cv2.getTrackbarPos("Smooth %", CTRL_WIN) / 100.0
        alpha = clamp(smooth_pct, 0.0, 0.99)
 
        filter_in_zone = cv2.getTrackbarPos("Filter in Zone", CTRL_WIN) == 1
        ttl = cv2.getTrackbarPos("Keep Boxes TTL", CTRL_WIN)
        tracker.ttl = max(1, ttl)
 
        # --- Build a perfectly aligned (symmetric) trapezoid ---
        cx = int(w * cx_pct)
        y_top = int(h * topy_pct)
        y_bot = int(h * boty_pct)
        top_half = int((w * topw_pct) / 2.0)
        bot_half = int((w * botw_pct) / 2.0)
 
        # clamp so corners stay on-screen
        top_half = min(top_half, cx - 1, w - cx - 1)
        bot_half = min(bot_half, cx - 1, w - cx - 1)
        y_top = clamp(y_top, 0, h - 2)
        y_bot = clamp(y_bot, y_top + 1, h - 1)
 
        zone = np.array([
            [cx - bot_half, y_bot],
            [cx - top_half, y_top],
            [cx + top_half, y_top],
            [cx + bot_half, y_bot],
        ], dtype=np.int32)
 
        # --- Lane detection inside the same zone ---
        left_line, right_line, _edges = detect_lane_lines(
            frame, zone, canny1, canny2, hth, minL, maxG
        )
 
        # Smooth endpoints (EMA) so they don’t jump
        if left_line is not None:
            prev_left = left_line if prev_left is None else tuple(int(ema(p, c, alpha)) for p, c in zip(prev_left, left_line))
        if right_line is not None:
            prev_right = right_line if prev_right is None else tuple(int(ema(p, c, alpha)) for p, c in zip(prev_right, right_line))
 
        # --- DNN detections ---
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        if fps_count % 2 == 0:
            dets = net.forward()
        else:
            dets = np.zeros((1, 1, 0, 7))
 
        detections = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < args.conf:
                continue
            idx = int(dets[0, 0, i, 1])
 
            # MobileNetSSD class IDs (VOC)
            # 15=person, 7=car, 6=bus, 14=motorbike, 2=bicycle
            label_map = {15: "person", 7: "car", 6: "bus", 14: "motorbike", 2: "bicycle"}
            if idx not in label_map:
                continue
            cls = label_map[idx]
 
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1)
            x2, y2 = clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                continue
 
            # filter cars to “in-zone” if enabled
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2
            inside = point_in_poly((cx_box, cy_box), zone)
 
            if filter_in_zone and cls in ("car", "bus", "motorbike", "bicycle") and not inside:
                continue
 
            detections.append({"box": (x1, y1, x2, y2), "cls": cls, "conf": conf})
 
        tracks = tracker.update(detections)
 
        # --- Draw overlay ---
        out = frame.copy()
 
        # green fill zone + cyan outline
        overlay = out.copy()
        cv2.fillPoly(overlay, [zone], (0, 255, 0))
        out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
 
        cv2.polylines(out, [zone], True, (255, 255, 0), 4)  # cyan-ish
 
        # Draw smoothed lanes (yellow)
        if prev_left is not None:
            x1, y1, x2, y2 = prev_left
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 6)
        if prev_right is not None:
            x1, y1, x2, y2 = prev_right
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 6)
 
        # center red marker
        cv2.line(out, (cx, y_bot), (cx, int((y_top + y_bot) / 2)), (0, 0, 255), 8)
 
        # Draw tracked boxes
        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            cls = t["cls"]
            conf = t["conf"]
            tid = t["id"]
 
            color = (255, 255, 255)
            if cls == "person":
                color = (255, 0, 255)
            elif cls in ("car", "bus", "motorbike", "bicycle"):
                color = (255, 255, 255)
 
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, f"{cls}:{conf:.2f} id{tid}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
 
        # FPS
        fps_count += 1
        dt = time.time() - fps_t0
        if dt >= 0.5:
            fps = fps_count / dt
            fps_t0 = time.time()
            fps_count = 0
        cv2.putText(out, f"FPS {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
 
        cv2.imshow("HUD", out)
 
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
 