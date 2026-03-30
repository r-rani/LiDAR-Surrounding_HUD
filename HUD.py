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
    return cv2.pointPolygonTest(poly.astype(np.int32), (float(pt[0]), float(pt[1])), False) >= 0

def ema(prev, cur, alpha):
    if prev is None:
        return cur
    return (alpha * prev) + ((1.0 - alpha) * cur)

def apply_zoom(img, zoom_factor):
    """
    Crops the center of the image and resizes it back to original size.
    zoom_factor 1.0 = no zoom.
    """
    if zoom_factor <= 1.0:
        return img
    
    h, w = img.shape[:2]
    # Calculate new dimensions
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    
    # Calculate crop boundaries (centered)
    y1 = (h - new_h) // 2
    y2 = y1 + new_h
    x1 = (w - new_w) // 2
    x2 = x1 + new_w
    
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

# -----------------------------
# Simple tracker to keep boxes briefly
# -----------------------------
class SimpleTracker:
    def __init__(self, iou_thresh=0.3, ttl=8):
        self.iou_thresh = iou_thresh
        self.ttl = ttl
        self.next_id = 1
        self.tracks = {}

    def update(self, dets):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["ttl"] -= 1
            if self.tracks[tid]["ttl"] <= 0:
                del self.tracks[tid]

        used = set()
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, d in enumerate(dets):
                if j in used or d["cls"] != tr["cls"]:
                    continue
                val = iou(tr["box"], d["box"])
                if val > best_iou:
                    best_iou = val
                    best_j = j

            if best_j != -1 and best_iou >= self.iou_thresh:
                d = dets[best_j]
                used.add(best_j)
                self.tracks[tid] = {
                    "box": d["box"], "cls": d["cls"],
                    "conf": d["conf"], "ttl": self.ttl,
                }

        for j, d in enumerate(dets):
            if j in used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": d["box"], "cls": d["cls"],
                "conf": d["conf"], "ttl": self.ttl,
            }

        out = []
        for tid, tr in self.tracks.items():
            out.append({"id": tid, **tr})
        return out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--prototxt", type=str, default="MobileNetSSD_deploy.prototxt")
    ap.add_argument("--model", type=str, default="MobileNetSSD_deploy.caffemodel")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return

    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    CTRL_WIN = "HUD Controls"
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 520, 500)

    cv2.createTrackbar("Digital Zoom x10", CTRL_WIN, 33, 40, lambda x: None)
    cv2.createTrackbar("Zone Center X %", CTRL_WIN, 50, 100, lambda x: None)
    cv2.createTrackbar("Zone Top Y %",    CTRL_WIN, 55, 95,  lambda x: None)
    cv2.createTrackbar("Zone Bottom Y %", CTRL_WIN, 92, 99,  lambda x: None)
    cv2.createTrackbar("Zone Top W %",    CTRL_WIN, 22, 90,  lambda x: None)
    cv2.createTrackbar("Zone Bot W %",    CTRL_WIN, 92, 100, lambda x: None)
    cv2.createTrackbar("Filter in Zone",  CTRL_WIN, 1, 1,    lambda x: None)
    cv2.createTrackbar("Keep Boxes TTL",  CTRL_WIN, 8, 30,   lambda x: None)

    tracker = SimpleTracker(iou_thresh=0.25, ttl=8)
    fps_t0, fps_count, fps = time.time(), 0, 0.0

    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
    
    while True:
        for _ in range(args.skip): cap.grab()
        ok, frame = cap.read()
        if not ok: break

        # 1. Digital Zoom
        zoom_val = cv2.getTrackbarPos("Digital Zoom x10", CTRL_WIN) / 10.0
        frame = apply_zoom(frame, zoom_val)
        
        # Ensure consistent size for processing
        frame = cv2.resize(frame, (1920, 1080))
        h, w = frame.shape[:2]

        # 2. Read sliders
        cx_pct = cv2.getTrackbarPos("Zone Center X %", CTRL_WIN) / 100.0
        topy_pct = cv2.getTrackbarPos("Zone Top Y %", CTRL_WIN) / 100.0
        boty_pct = cv2.getTrackbarPos("Zone Bottom Y %", CTRL_WIN) / 100.0
        topw_pct = cv2.getTrackbarPos("Zone Top W %", CTRL_WIN) / 100.0
        botw_pct = cv2.getTrackbarPos("Zone Bot W %", CTRL_WIN) / 100.0

        # 3. Trapezoid setup
        cx = int(w * cx_pct)
        y_top, y_bot = int(h * topy_pct), int(h * boty_pct)
        top_half = min(int((w * topw_pct) / 2.0), cx - 1, w - cx - 1)
        bot_half = min(int((w * botw_pct) / 2.0), cx - 1, w - cx - 1)

        zone = np.array([[cx - bot_half, y_bot], [cx - top_half, y_top],
                         [cx + top_half, y_top], [cx + bot_half, y_bot]], dtype=np.int32)

        # 4. DNN detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        dets = net.forward() if fps_count % 2 == 0 else np.zeros((1, 1, 0, 7))

        detections = []
        label_map = {15: "person", 7: "car", 6: "bus", 14: "motorbike", 2: "bicycle"}
        for i in range(dets.shape[2]):
            conf, idx = float(dets[0, 0, i, 2]), int(dets[0, 0, i, 1])
            if conf < args.conf or idx not in label_map: continue
            
            box = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1, x2, y2 = [clamp(v, 0, w-1 if i%2==0 else h-1) for i, v in enumerate(box)]
            
            if cv2.getTrackbarPos("Filter in Zone", CTRL_WIN) == 1:
                if not point_in_poly(((x1+x2)//2, (y1+y2)//2), zone): continue
            
            detections.append({"box": (x1, y1, x2, y2), "cls": label_map[idx], "conf": conf})

        tracks = tracker.update(detections)

        # 5. Rendering
        out = np.zeros_like(frame)
        overlay = out.copy()
        cv2.fillPoly(overlay, [zone], (0, 255, 0))
        out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
        cv2.polylines(out, [zone], True, (255, 255, 0), 4)

        cv2.line(out, (cx, y_bot), (cx, int((y_top + y_bot) / 2)), (0, 0, 255), 8)

        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            color = (255, 0, 255) if t["cls"] == "person" else (255, 255, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, f"{t['cls']}:{t['conf']:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_count += 1
        if (time.time() - fps_t0) >= 0.5:
            fps = fps_count / (time.time() - fps_t0)
            fps_t0, fps_count = time.time(), 0
        cv2.putText(out, f"FPS {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

        flipped = cv2.flip(out, 1)   # horizontal flip (HUD mode)
        cv2.imshow("HUD", flipped)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
