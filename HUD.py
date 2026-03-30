#!/usr/bin/env python3

import argparse
import time

import cv2
import numpy as np


# -----------------------------
# Helpers
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

    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / float(area_a + area_b - inter + 1e-9)


# -----------------------------
# Simple tracker
# Keeps boxes alive briefly so they do not flicker
# -----------------------------
class SimpleTracker:
    def __init__(self, iou_thresh=0.25, ttl=8):
        self.iou_thresh = iou_thresh
        self.ttl = ttl
        self.next_id = 1
        self.tracks = {}

    def update(self, dets):
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["ttl"] -= 1
            if self.tracks[tid]["ttl"] <= 0:
                del self.tracks[tid]

        used = set()

        # Match detections to existing tracks
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

        # Add new tracks
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

        out = []
        for tid, tr in self.tracks.items():
            out.append({
                "id": tid,
                "box": tr["box"],
                "cls": tr["cls"],
                "conf": tr["conf"],
            })
        return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0, help="Camera index")
    ap.add_argument("--video", type=str, default=None, help="Optional video file")
    ap.add_argument("--prototxt", type=str, default="MobileNetSSD_deploy.prototxt")
    ap.add_argument("--model", type=str, default="MobileNetSSD_deploy.caffemodel")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--skip", type=int, default=1, help="Frames to skip between reads")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    # Open source
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return

    # Load DNN
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    # Window
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty("HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracker = SimpleTracker(iou_thresh=0.25, ttl=8)

    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0

    # MobileNet SSD classes of interest
    label_map = {
        15: "person",
        7: "car",
        6: "bus",
        14: "motorbike",
        2: "bicycle",
    }

    detect_every = 2
    cached_detections = []

    while True:
        # Skip frames to improve FPS
        for _ in range(args.skip):
            cap.grab()

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (args.width, args.height))
        h, w = frame.shape[:2]

        # Run DNN every N frames
        if fps_count % detect_every == 0:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                scalefactor=0.007843,
                size=(300, 300),
                mean=127.5
            )
            net.setInput(blob)
            dets = net.forward()

            detections = []
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                idx = int(dets[0, 0, i, 1])

                if conf < args.conf:
                    continue
                if idx not in label_map:
                    continue

                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                x1 = clamp(x1, 0, w - 1)
                y1 = clamp(y1, 0, h - 1)
                x2 = clamp(x2, 0, w - 1)
                y2 = clamp(y2, 0, h - 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "cls": label_map[idx],
                    "conf": conf
                })

            cached_detections = detections

        tracks = tracker.update(cached_detections)

        # -----------------------------
        # BLACK HUD BACKGROUND
        # -----------------------------
        out = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw only object boxes
        for t in tracks:
            x1, y1, x2, y2 = t["box"]

            if t["cls"] == "person":
                color = (255, 0, 255)   # purple
            else:
                color = (0, 255, 0)     # green

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{t['cls']} {t['conf']:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # FPS
        fps_count += 1
        now = time.time()
        if (now - fps_t0) >= 0.5:
            fps = fps_count / (now - fps_t0)
            fps_t0 = now
            fps_count = 0

        cv2.putText(
            out,
            f"FPS {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Flip for HUD reflection
        flipped = cv2.flip(out, 1)

        cv2.imshow("HUD", flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()