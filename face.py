# face.py  — headless-safe version (no cv2.imshow)
import cv2
import os
import time
import json
import sqlite3
import logging
from insightface.app import FaceAnalysis
import numpy as np

# ---------------- CONFIG ----------------
CONFIG_FILE = "config.json"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {"detection_skip_frames": 5, "log_folder": "logs", "save_frames_every": 30}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

DETECTION_SKIP = int(config.get("detection_skip_frames", 5))
LOG_FOLDER = config.get("log_folder", "logs")
SAVE_FRAMES_EVERY = int(config.get("save_frames_every", 30))
os.makedirs(LOG_FOLDER, exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "events.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- DATABASE ----------------
DB_PATH = os.path.join("database", "faces.db")
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS visitors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER,
    timestamp TEXT,
    event TEXT,
    image_path TEXT
)
""")
conn.commit()

# ---------------- INSIGHTFACE ----------------
# IMPORTANT: set model_root to where you extracted buffalo_l (or other models)
# Example: model_root = r"C:\Users\sriga\OneDrive\New folder\face tracker\models"
# If you put models folder inside project, use: model_root = os.path.abspath("models")
model_root = os.path.abspath("models")
print("Using model root:", model_root)

app = FaceAnalysis(allowed_modules=['detection', 'recognition'], root=model_root)
app.prepare(ctx_id=-1)  # newer versions don't accept nms arg

# ---------------- TRACKING ----------------
face_database = {}        # {face_id: embedding}
active_faces = {}         # {face_id: last_seen_frame_index}
unique_count = 0
frame_count = 0
EXIT_THRESHOLD = 20       # frames to consider face exited

# ---------------- OUTPUT FOLDERS ----------------
entries_folder = os.path.join(LOG_FOLDER, "entries")
exits_folder = os.path.join(LOG_FOLDER, "exits")
frames_folder = os.path.join(LOG_FOLDER, "frames")
os.makedirs(entries_folder, exist_ok=True)
os.makedirs(exits_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

# ---------------- CAMERA / VIDEO ----------------
# Use 0 for default webcam, or put path "video.mp4" / RTSP URL
VIDEO_SOURCE = 0
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

print("Started capture. Press Ctrl+C in the terminal to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame returned — end of stream or camera problem.")
            break

        frame_count += 1

        # Save a periodic frame to disk for quick visual inspection (headless)
        if frame_count % SAVE_FRAMES_EVERY == 0:
            fname = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(fname, frame)
            logging.info(f"Saved frame for inspection: {fname}")

        # Skip heavy detection on some frames to save CPU
        if frame_count % DETECTION_SKIP != 0:
            continue

        faces = app.get(frame)  # list of face objects
        current_face_ids = []

        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # sanity clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

            # Compare to known embeddings
            match_id = None
            for fid, femb in face_database.items():
                if np.linalg.norm(emb - femb) < 0.6:  # threshold (tune if needed)
                    match_id = fid
                    break

            # New face → register
            if match_id is None:
                unique_count += 1
                match_id = unique_count
                face_database[match_id] = emb

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                crop = frame[y1:y2, x1:x2] if (y2>y1 and x2>x1) else frame
                img_path = os.path.join(entries_folder, f"face_{match_id}_{timestamp}.jpg")
                cv2.imwrite(img_path, crop)

                # DB + log
                c.execute(
                    "INSERT INTO visitors(face_id, timestamp, event, image_path) VALUES (?, ?, ?, ?)",
                    (match_id, timestamp, "entry", img_path)
                )
                conn.commit()
                logging.info(f"New face registered: ID={match_id}, image={img_path}")
                print(f"[ENTRY] ID={match_id} saved {img_path}")

            else:
                # recognized re-entry or seen again; optionally log
                logging.info(f"Recognized face: ID={match_id}")
                # For entry events on reappearance you may also want to record an "entry" if you require that

            # update tracking info
            active_faces[match_id] = frame_count
            current_face_ids.append(match_id)

        # Detect exits: if a tracked face was not seen for EXIT_THRESHOLD frames -> exit
        exited = []
        for fid, last_seen in list(active_faces.items()):
            if fid not in current_face_ids:
                if frame_count - last_seen > EXIT_THRESHOLD:
                    exited.append(fid)

        for fid in exited:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Save last-seen full frame (or you can save crop from last known box if stored)
            img_path = os.path.join(exits_folder, f"face_{fid}_{timestamp}.jpg")
            cv2.imwrite(img_path, frame)

            c.execute(
                "INSERT INTO visitors(face_id, timestamp, event, image_path) VALUES (?, ?, ?, ?)",
                (fid, timestamp, "exit", img_path)
            )
            conn.commit()
            logging.info(f"Face exited: ID={fid}, image={img_path}")
            print(f"[EXIT] ID={fid} saved {img_path}")
            del active_faces[fid]

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C). Shutting down...")

finally:
    cap.release()
    conn.close()
    print("Stopped. All resources released.")
