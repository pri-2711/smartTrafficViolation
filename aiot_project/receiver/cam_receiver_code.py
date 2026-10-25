import socket
import os
import sys
import queue
import threading
from datetime import datetime
import requests

# AI project lives in a sibling folder; resolve it relative to this file
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "Traffic-Violation-main"
    )
)

from pipeline.aggregator import Pipeline

# =========================
# SETTINGS
# =========================

CAMERA_IP = "10.244.95.144"
CAPTURE_URL = f"http://{CAMERA_IP}/capture"

RECEIVER_PORT = 5000
SAVE_FOLDER = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "snapshots"
    )
)

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================
# START RECEIVER
# =========================

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind(("0.0.0.0", RECEIVER_PORT))
server.listen(5)

print(f"Receiver listening on port {RECEIVER_PORT}")
print("SYSTEM READY")

snapshot_number = 0

print("Loading AI pipeline...")
pipeline = Pipeline()
print("AI pipeline ready")

# Snapshots are analysed one at a time on a worker thread, so a slow
# process() call never blocks the TCP server or the camera requests.
analysis_queue = queue.Queue(maxsize=5)


def analysis_worker():

    while True:

        filepath, speed_kmph = analysis_queue.get()

        try:
            record = pipeline.process(filepath, speed_kmph=speed_kmph)

            print()
            print("Analysed:", filepath)
            print(
                "Violation types:",
                ", ".join(record["violation_types"]) or "none"
            )
            print(
                "Plate number:",
                record["plate_number"] or "(none read)"
            )

        except Exception as e:

            print("PIPELINE ERROR:")
            print(e)

        finally:

            analysis_queue.task_done()


threading.Thread(target=analysis_worker, daemon=True).start()

# =========================
# WAIT FOR ESP32 DEVKIT
# =========================

while True:

    client, address = server.accept()

    try:
        data = client.recv(1024).decode().strip()

        print()
        print("Trigger received:")
        print(data)

        if data.startswith("CAPTURE|"):

            parts = data.split("|")

            speed_kmph = None
            if len(parts) >= 4:
                try:
                    speed_kmph = float(parts[3])
                except ValueError:
                    speed_kmph = None

            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
            else:
                now = datetime.now()
                date_part = now.strftime("%d_%m_%Y")
                time_part = now.strftime("%H_%M_%S")

            # ==========================================
            # REQUEST ONE FRAME FROM ESP32-CAM
            # ==========================================

            print("Requesting image from ESP32-CAM...")

            response = requests.get(
                CAPTURE_URL,
                timeout=10
            )

            if response.status_code == 200 and response.content:

                snapshot_number += 1

                filename = (
                    f"snapshot{snapshot_number}_"
                    f"{date_part}_{time_part}.jpg"
                )

                filepath = os.path.join(
                    SAVE_FOLDER,
                    filename
                )

                with open(filepath, "wb") as f:
                    f.write(response.content)

                print("IMAGE SAVED:")
                print(filepath)

                try:
                    analysis_queue.put_nowait((filepath, speed_kmph))
                    print("Queued for analysis")

                except queue.Full:

                    print("Analysis queue full - snapshot kept, not analysed")

            else:

                print("ERROR: ESP32-CAM did not return an image.")
                print("HTTP status:", response.status_code)

    except requests.exceptions.RequestException as e:

        print("CAMERA ERROR:")
        print(e)

    except Exception as e:

        print("Receiver error:")
        print(e)

    finally:

        client.close()