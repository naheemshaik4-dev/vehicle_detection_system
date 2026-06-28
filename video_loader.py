"""
Vehicle Detection & Counter using YOLOv8 + OpenCV
--------------------------------------------------
Requirements:
    pip install ultralytics opencv-python

Usage:
    python vehicle_counter.py                         # uses webcam (source=0)
    python vehicle_counter.py --source video.mp4      # uses a video file
    python vehicle_counter.py --source video.mp4 --output out.mp4  # saves result
"""

import cv2
from ultralytics import YOLO

# Input video source in the workspace root
SOURCE_VIDEO = "traffic.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
SHOW = True

# ── Overlay appearance ─────────────────────────────────────────────────────────
BOX_COLOR   = (0, 200, 0)      # green bounding box
TEXT_COLOR  = (255, 255, 255)  # white label text
PANEL_COLOR = (20, 20, 20)     # dark HUD panel
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2

# ── COCO class IDs that count as vehicles ──────────────────────────────────────
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}


def draw_hud(frame, counts: dict, total: int) -> None:
    """Draw a semi-transparent HUD panel with the total vehicle count."""
    line = f"  TOTAL VEHICLES: {total}"

    line_h   = 26
    pad      = 12
    panel_w  = 260
    panel_h  = line_h + pad * 2 + 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Vehicle Count", (18, 34),
                FONT, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line, (18, 34 + 28), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


def process_video(source, model_name: str, conf: float, show: bool) -> None:
    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)

    # ── Open video source ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {source}  |  {width}×{height} @ {fps:.1f} fps")

    # ── Output writer, save next to the input video with a suffix ──────────────
    writer = None
    output_path = None
    if isinstance(source, str) and "." in source:
        ext = source.rsplit('.', 1)[1].lower()
        output_path = source.rsplit('.', 1)[0] + f"_counted.{ext}"
    else:
        output_path = str(source) + "_counted.mp4"

    if output_path:
        if ext == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif ext in {"avi", "mov"}:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            print(f"[WARN] Unknown output extension '.{ext}', defaulting to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {output_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Run YOLOv8 inference ────────────────────────────────────────────────
        results = model(frame, conf=conf, verbose=False)[0]

        per_class_count = {label: 0 for label in VEHICLE_CLASSES.values()}
        total = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            label = VEHICLE_CLASSES[cls_id]
            per_class_count[label] += 1
            total += 1

        # ── HUD overlay ─────────────────────────────────────────────────────────
        draw_hud(frame, per_class_count, total)

        # frame counter (bottom-right)
        fc_text = f"Frame {frame_idx}"
        (fw, fh), _ = cv2.getTextSize(fc_text, FONT, 0.45, 1)
        cv2.putText(frame, fc_text, (width - fw - 10, height - 10),
                    FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Vehicle Counter — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit requested.")
                break

    # ── Cleanup ─────────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frame(s).")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_video(
        source     = SOURCE_VIDEO,
        model_name = MODEL_NAME,
        conf       = CONFIDENCE,
        show       = SHOW,
    )