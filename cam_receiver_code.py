import socket
import os
from datetime import datetime
import requests

# =========================
# SETTINGS
# =========================

CAMERA_IP = "10.136.145.144"
CAPTURE_URL = f"http://{CAMERA_IP}/capture"

RECEIVER_PORT = 5000
SAVE_FOLDER = "snapshots"

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