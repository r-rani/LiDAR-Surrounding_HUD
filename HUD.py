import cv2
import numpy as np

# =========================
# SETTINGS
# =========================
SCREEN_W = 1920
SCREEN_H = 1080

DISPLAY_X = 0
DISPLAY_Y = 0

WINDOW_NAME = "HUD"

MIRROR = True
SMOOTHING = 0.7   # higher = smoother, slower

# Load person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Previous box for smoothing
prev_box = None


# =========================
# FUNCTIONS
# =========================
def smooth_box(prev, new):
    if prev is None:
        return new
    px, py, pw, ph = prev
    nx, ny, nw, nh = new

    sx = int(px * SMOOTHING + nx * (1 - SMOOTHING))
    sy = int(py * SMOOTHING + ny * (1 - SMOOTHING))
    sw = int(pw * SMOOTHING + nw * (1 - SMOOTHING))
    sh = int(ph * SMOOTHING + nh * (1 - SMOOTHING))

    return (sx, sy, sw, sh)


def draw_hud_box(frame, x1, y1, x2, y2):
    color = (255, 0, 255)

    # Glow effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 12)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)

    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


# =========================
# WINDOW SETUP
# =========================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, DISPLAY_X, DISPLAY_Y)
cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# =========================
# MAIN LOOP
# =========================
while True:
    ret, cam = cap.read()
    if not ret:
        break

    if MIRROR:
        cam = cv2.flip(cam, 1)

    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

    # Detect people
    boxes, _ = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Create black HUD canvas
    hud = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    if len(boxes) > 0:
        # Take largest detected person
        largest = max(boxes, key=lambda b: b[2] * b[3])
        x, y, w, h = largest

        # Smooth movement
        smooth = smooth_box(prev_box, (x, y, w, h))
        prev_box = smooth

        x, y, w, h = smooth

        # Scale camera coords to screen
        scale_x = SCREEN_W / cam.shape[1]
        scale_y = SCREEN_H / cam.shape[0]

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        draw_hud_box(hud, x1, y1, x2, y2)

    cv2.imshow(WINDOW_NAME, hud)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()