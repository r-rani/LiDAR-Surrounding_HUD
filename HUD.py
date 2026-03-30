#!/usr/bin/env python3
import cv2
import numpy as np
import time
import math

# =========================================================
# SETTINGS
# =========================================================

# Change these to match the monitor/projector resolution
SCREEN_W = 1920
SCREEN_H = 1080

# If the HUD monitor is the ONLY display, leave these at 0,0
# If using extended desktop and the HUD monitor is to the right
# of a 1920x1080 laptop screen, set DISPLAY_X = 1920, DISPLAY_Y = 0
DISPLAY_X = 0
DISPLAY_Y = 0

WINDOW_NAME = "HUD"

# HUD box settings
BOX_W = 420
BOX_H = 260
BOX_COLOR = (255, 0, 255)   # magenta in BGR
BOX_THICKNESS = 3
GLOW = True

# Set True if you want the box centered
CENTER_BOX = True

# If CENTER_BOX = False, these are used
BOX_X = 700
BOX_Y = 400

# Optional crosshair
DRAW_CENTER_DOT = False

# Optional FPS text
SHOW_FPS = False

# =========================================================
# DRAW HELPERS
# =========================================================

def draw_glow_rect(img, pt1, pt2, color):
    if GLOW:
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, 14)
        cv2.addWeighted(overlay, 0.05, img, 0.95, 0, img)

        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, 8)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, 4)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    cv2.rectangle(img, pt1, pt2, color, BOX_THICKNESS)

def draw_corner_markers(img, x1, y1, x2, y2, color, length=32, thickness=2):
    # top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    # top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    # bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    # bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def get_box_coords():
    if CENTER_BOX:
        x1 = (SCREEN_W - BOX_W) // 2
        y1 = (SCREEN_H - BOX_H) // 2
    else:
        x1 = BOX_X
        y1 = BOX_Y

    x2 = x1 + BOX_W
    y2 = y1 + BOX_H

    # clamp
    x1 = max(0, min(x1, SCREEN_W - 1))
    y1 = max(0, min(y1, SCREEN_H - 1))
    x2 = max(0, min(x2, SCREEN_W - 1))
    y2 = max(0, min(y2, SCREEN_H - 1))

    return x1, y1, x2, y2

# =========================================================
# MAIN
# =========================================================

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Make it borderless/fullscreen
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Move it onto the HUD monitor
    cv2.moveWindow(WINDOW_NAME, DISPLAY_X, DISPLAY_Y)

    prev = time.time()

    while True:
        # Full black canvas
        frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        x1, y1, x2, y2 = get_box_coords()

        draw_glow_rect(frame, (x1, y1), (x2, y2), BOX_COLOR)
        draw_corner_markers(frame, x1, y1, x2, y2, BOX_COLOR)

        if DRAW_CENTER_DOT:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, BOX_COLOR, -1)

        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()