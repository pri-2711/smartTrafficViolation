# Smart Traffic Violation Detection (AIoT, CSE712)

A small checkpoint system for two-wheelers. Sensors notice a vehicle, a camera takes pictures,
and an AI reads each picture to find **helmet violations**, **triple riding**, **overspeeding**
and the **number plate**. Every result is saved to a database and shown on a dashboard.

---

## 1. The idea in one minute

1. Someone drives past the checkpoint.
2. Motion sensors (PIR) see them. An ultrasonic sensor measures how fast they are going.
3. If the speed is above the limit, a buzzer sounds.
4. While motion continues, the system keeps taking pictures with the camera.
5. Each picture goes to a PC, where AI models look for violations and read the plate.
6. The result is saved (database + an annotated picture) and appears on the dashboard.

## 2. How it all fits together

```
 +---------------------------- HARDWARE ----------------------------+
 |                                                                  |
 |   ESP32 DevKit                              ESP32-CAM            |
 |   - 2 PIR motion sensors                    - takes JPEG photos  |
 |   - ultrasonic distance sensor              - 1600x1200          |
 |   - DS3231 clock (RTC)                      - serves a photo at  |
 |   - buzzer                                    http://<cam>/capture
 |        |                                          ^              |
 +--------|------------------------------------------|--------------+
          | (1) Wi-Fi, TCP port 5000                 | (3) Wi-Fi, HTTP GET
          |     "CAPTURE|date|time|speed"             |
          v                                          |
 +---------------------------- PC (Python) ----------------------------+
 |                                                                      |
 |   receiver  (cam_receiver_code.py)                                   |
 |     (2) reads the message, (3) asks the camera for a photo,          |
 |     (4) saves it and puts it in a queue                              |
 |                          |                                           |
 |                          v                                           |
 |   AI pipeline  (pipeline/aggregator.py)  - runs in a background thread
 |     (5) helmet / no-helmet / plate model                             |
 |         triple-riding model                                          |
 |         plate reading (OCR)                                          |
 |         speed check (over 40 km/h?)                                  |
 |                          |                                           |
 |                          v                                           |
 |     (6) saves:  violations.db  +  violations/evidence/*.png          |
 |                          |                                           |
 |                          v                                           |
 |   dashboard  (Streamlit, http://localhost:8501)  <- reads the database
 +----------------------------------------------------------------------+
```

The DevKit and the camera never talk to each other directly. The PC is in the middle.

## 3. Step-by-step flow

| Step | Who | What happens |
|---|---|---|
| 1 | DevKit | Waits 30 s after power-on so the PIR sensors can settle. |
| 2 | DevKit | A PIR sensor sees motion, so it starts measuring speed with the ultrasonic sensor. |
| 3 | DevKit | Every 150 ms it measures distance. Speed = how much the distance shrank between two readings. |
| 4 | DevKit | If speed is above **40 km/h**, the buzzer sounds for 3 s, then a 5 s cooldown. |
| 5 | DevKit | 1 s after motion starts, and then every 1.5 s while motion continues, it sends a message to the PC: `CAPTURE\|DD_MM_YYYY\|HH_MM_SS\|speed`. Speed is left out if none has been measured yet. |
| 6 | Receiver | Reads the message and asks the camera for a photo (`/capture`). Saves it in `aiot_project/snapshots/`. |
| 7 | Receiver | Puts the photo in a queue (up to 5). A background worker analyses one photo at a time, so slow AI never blocks new messages. |
| 8 | AI pipeline | Runs the helmet/plate model and the triple-riding model on the photo. |
| 9 | AI pipeline | If a plate was found, crops it and reads the text with OCR. |
| 10 | AI pipeline | Marks `no_helmet`, `triple_riding` and/or `overspeed` (speed over 40). |
| 11 | AI pipeline | Draws all boxes on one evidence image and adds a row to `violations.db`. |
| 12 | Dashboard | Shows the log and lets you try an image by hand. |

## 4. Tech stack

**Hardware**

| Part | Used for |
|---|---|
| ESP32 DevKit | Brain of the sensor side |
| ESP32-CAM (AI-Thinker) + ESP32-CAM-MB base | Camera, flashed over USB |
| 2 x PIR sensors | Detect motion |
| HC-SR04 ultrasonic sensor | Measure distance, so speed |
| DS3231 RTC | Real-time clock for timestamps |
| Buzzer (active-low) | Overspeed alarm |

**Firmware (C++ / Arduino)**: ESP32 Arduino core 3.3, `WiFi`, `esp_camera`, `RTClib`. Built and
flashed with `arduino-cli`.

**PC software (Python 3.12)**

| Tool | Used for |
|---|---|
| Python `socket` + `threading` + `queue` | Receiving messages and analysing in the background |
| `requests` | Fetching photos from the camera |
| PyTorch | Runs the AI models |
| Ultralytics **YOLO26** | The two detection models used in the live system |
| Custom "Option B" detector (MobileNetV3-Small) | Our own model built for the course; trained and compared, not used live |
| EasyOCR | Reads the number plate text |
| OpenCV + NumPy + Pillow | Image handling and drawing boxes |
| SQLite | The violation log (`violations.db`) |
| Streamlit + pandas | The dashboard |

**The two AI models**

| Model | Finds |
|---|---|
| Helmet and plate | `bike`, `helmet`, `no-helmet`, `number-plate` |
| Triple riding | `tripleriding` |

## 5. Folder layout

```
AIOT/
├── run_system.py              <- START HERE: runs everything
├── README.md                  <- this file
├── .venv/                     Python environment (all packages installed)
│
├── aiot_project/              the hardware side
│   ├── firmware/
│   │   ├── trafficViolationSystem/   DevKit code (sensors, buzzer, sends triggers)
│   │   ├── CameraWebServer/          ESP32-CAM code (serves photos)
│   │   └── prototypes/               early test sketches for each part
│   ├── receiver/cam_receiver_code.py receives triggers, fetches photos, queues AI
│   ├── snapshots/                    every photo the camera took
│   └── docs/                         circuit diagram and wiring notes
│
└── Traffic-Violation-main/    the AI side
    ├── pipeline/aggregator.py  combines models + OCR + database
    ├── ocr/plate_ocr.py        plate reading helpers
    ├── dashboard/app.py        the Streamlit dashboard
    ├── training/               training and evaluation code
    │   └── runs/               trained model files (best.pt)
    ├── dataset/                only small class-name files here (full dataset not included)
    ├── violations.db           the violation log (created while running)
    ├── violations/evidence/    annotated pictures (created while running)
    └── README.md               more detail on the AI work
```

## 6. How to run it

**Start everything with one command:**

```powershell
cd D:\university\year4\AIOT
.\.venv\Scripts\python.exe run_system.py
```

It checks your setup, then starts the receiver (with the AI) and the dashboard.
Open **http://localhost:8501** for the dashboard. Press **Ctrl+C** to stop everything.

Other options:

```powershell
.\.venv\Scripts\python.exe run_system.py --check          # only check the setup
.\.venv\Scripts\python.exe run_system.py --no-dashboard   # receiver + AI only
```

**Before you start, make sure:**
- The PC, the DevKit and the ESP32-CAM are all on the **same Wi-Fi**.
- Both boards are powered on.
- The firewall allows TCP port 5000 (already set up on this PC).

**Test the AI on a single picture without hardware:**

```powershell
.\.venv\Scripts\python.exe Traffic-Violation-main\pipeline\aggregator.py --image some_photo.jpg --speed 55
```

## 7. Settings you may need to change

| Setting | Where | Current value |
|---|---|---|
| Speed limit (must match in both places) | `OVERSPEED_THRESHOLD_KMPH` in `trafficViolationSystem.ino` and `pipeline/aggregator.py` | 40 km/h |
| PC address the DevKit sends to | `PYTHON_SERVER_IP` in `trafficViolationSystem.ino` | PC's IP (check with `ipconfig`) |
| Camera address | `CAMERA_IP` in `receiver/cam_receiver_code.py` | Camera's IP (shown on its serial monitor) |
| Time between pictures | `CAPTURE_INTERVAL_MS` in `trafficViolationSystem.ino` | 1500 ms |
| Delay before the first picture | `CAMERA_DELAY_MS` in `trafficViolationSystem.ino` | 1000 ms |
| Port | `RECEIVER_PORT` in `cam_receiver_code.py` and `PYTHON_SERVER_PORT` in the sketch | 5000 |

If the PC's IP changes (new Wi-Fi, restart), update `PYTHON_SERVER_IP` and re-flash the DevKit.
If the camera's IP changes, update `CAMERA_IP`. `run_system.py` warns you about both when it starts.

**Re-flashing a board** (close any Serial Monitor first, or the port is busy):

```powershell
$cli = 'C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe'
& $cli compile --fqbn esp32:esp32:esp32 aiot_project\firmware\trafficViolationSystem
& $cli upload -p COM3 --fqbn esp32:esp32:esp32 aiot_project\firmware\trafficViolationSystem
& $cli compile --fqbn esp32:esp32:esp32cam aiot_project\firmware\CameraWebServer
& $cli upload -p COM8 --fqbn esp32:esp32:esp32cam aiot_project\firmware\CameraWebServer
```

COM3 and COM8 are the DevKit and camera on this PC. Your ports may differ.

## 8. If something goes wrong

| Problem | Likely cause |
|---|---|
| DevKit says "Could not connect to Python server" | The receiver isn't running, the PC's IP changed, or the firewall blocks port 5000. |
| Receiver says "CAMERA ERROR" | Wrong `CAMERA_IP`, the camera is off, or it is on another Wi-Fi. |
| Camera is slow or shows black frames | Weak power (try a better USB cable or port), weak Wi-Fi, or poor light. |
| "Access is denied" when uploading | A Serial Monitor or other program is using the port. Close it. |
| Receiver exits when starting | Model files or the `dataset` class-name files are missing. |
| "Analysis queue full" | Photos arrive faster than the AI can check them (the AI runs on CPU). |

## 9. Known limits (please be honest about these in the demo)

- The AI runs on the **CPU**, so each picture takes a few seconds.
- **Plate reading accuracy has not been measured.** The OCR works, but there are too few clear Indian plate images to give a trustworthy number.
- The **40 km/h limit is a placeholder**. Re-measure real speeds on the final setup and update both places.
- Speed comes from a single ultrasonic sensor, so it is an estimate, not a calibrated reading.
- **Every analysed picture is logged**, not only violations, so the log grows quickly.
- The DS3231 clock was set from the computer's time when first programmed, so its date can drift or be off by a day. This only affects the photo file names.
- Our own "Option B" model scores lower than YOLO26. That is expected, because it is a small model built for learning.
- The Wi-Fi name and password are written inside the sketches. Do not share the code publicly without removing them.
