# ESP32-CAM web server

The `CameraWebServer` example from the Arduino esp32 core 3.3.12, adapted for
an **ESP32-CAM AI-Thinker** on an **MB** USB programmer board. The camera
module is labelled **RHYX M21-45**, which is a **GC2145** sensor. It has **no
hardware JPEG** encoder.

## Changes from the stock example

| Change | Where | Why |
|---|---|---|
| `#define CAMERA_MODEL_AI_THINKER` | `board_config.h` | Selects the AI-Thinker pin map. |
| `PIXFORMAT_RGB565` instead of `PIXFORMAT_JPEG` | `CameraWebServer.ino` | The GC2145 cannot output JPEG; asking for it fails camera init with error `0x106`. In RGB565 the example falls back to 240x240, and `/capture` JPEG-encodes in software. |
| WiFi credentials moved to `secrets.h` (gitignored) | `CameraWebServer.ino` includes `secrets.h` | Keeps the network password out of git. |
| Brownout detector disabled in `setup()` (`WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0)`) and WiFi TX power lowered to 8.5 dBm (`WiFi.setTxPower(WIFI_POWER_8_5dBm)`) | `CameraWebServer.ino` | Startup brownout resets on USB power. **This is a workaround:** the real fix is a better power supply (a stable 5 V source that can deliver the current spikes). |

### `secrets.h`

Create `CameraWebServer/secrets.h` (it is in `.gitignore`) with exactly two lines:

```cpp
const char *ssid = "your-2.4GHz-network";
const char *password = "your-password";
```

The ESP32 only joins 2.4 GHz networks.

## Build, flash, monitor

```sh
cd CameraWebServer
arduino-cli compile --fqbn esp32:esp32:esp32cam .
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32cam .
arduino-cli monitor -p /dev/ttyUSB0 --config baudrate=115200,dtr=off,rts=off
```

The monitor needs `dtr=off,rts=off`. With DTR/RTS asserted, the MB board's
auto-reset circuit holds the ESP32 in reset and nothing prints. The serial log
shows the camera's IP address; then open `http://<ip>/` for the web UI or
`http://<ip>/capture` for a single JPEG frame.

## Known limits

- 240x240 pixels in RGB565 only, since the sensor has no hardware JPEG.
- About 5–10 fps on the stream, because every frame is JPEG-encoded in software.

## PC side: face detection

The ESP32 only serves frames. Face detection runs on the PC, on the CPU, with
OpenCV's YuNet.

### Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r pc/requirements.txt    # opencv-python 5.0.0.93, numpy 2.5.3
```

OpenCV 5.0 includes `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF` in the
main `opencv-python` package (no contrib needed).

Download the models into `models/` (gitignored) from
[opencv_zoo](https://github.com/opencv/opencv_zoo), pinned to commit
`47534e27c9851bb1128ccc0102f1145e27f23f98`. The files are in Git LFS, so use
the `media.githubusercontent.com` URLs:

```sh
Z=https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models
mkdir -p models
curl -fL -o models/face_detection_yunet_2026may.onnx   $Z/face_detection_yunet/face_detection_yunet_2026may.onnx
curl -fL -o models/face_recognition_sface_2021dec.onnx $Z/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

| Model | File | Size | SHA-256 | License |
|---|---|---:|---|---|
| YuNet (face detection) | `face_detection_yunet_2026may.onnx` | 229,738 B | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` | MIT (Shiqi Yu) |
| SFace (face recognition, not used yet) | `face_recognition_sface_2021dec.onnx` | 38,696,353 B | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

`2026may` is the zoo's default YuNet for OpenCV 5. It is the `2023mar` model
re-exported with dynamic input dims, which OpenCV 5's ONNX Runtime engine needs
for arbitrary frame sizes.

### Run

```sh
CAM_IP=192.168.20.71 .venv/bin/python pc/capture_detect.py --seconds 30
.venv/bin/python pc/capture_detect.py --preview --warmup 10 --seconds 30   # watch it live
.venv/bin/python pc/capture_detect.py --seconds 60 --save   # keep frames that contain a face
```

Every second it fetches `http://$CAM_IP/capture` (default `192.168.20.71`),
runs YuNet (score ≥ 0.9, as in the zoo demo; change with `--score`), and prints
the timestamp, fetch time, detection time, number of faces, and each face's
score and box. At the end it prints the detection rate, mean score, and mean
fetch and detection times.

`--preview` shows each frame in a window, scaled up 2x, with the face box, its
score, and the running count of frames with a face. Press `q` to stop; the
window also closes when the run ends. `--warmup N` first shows N seconds
labelled WARM-UP (e.g. to position yourself) that are not measured or saved,
then continues straight into the measured `--seconds` in the same window. On
Wayland the script uses XWayland (`QT_QPA_PLATFORM=xcb`), since the
opencv-python wheel only ships Qt's X11 plugin. Qt prints harmless font
warnings on stderr.

`--save` writes only frames with at least one face, with the boxes drawn, to
`data/frames/` (gitignored). Frames without a face are never saved. No frames
or face images go into the repo (`docs/` and `data/` are both ignored).

Nothing is written to disk without `--save`.

### First test (2026-10-01)

One person in front of the camera, `--preview --warmup 10 --seconds 30`, score
threshold 0.9:

| Measure | Result |
|---|---|
| frames with ≥1 face | 20/30 (67%) |
| mean face score | 0.923 (range 0.917–0.929) |
| mean fetch time (`/capture`, 240x240) | 289.7 ms |
| mean YuNet detection time (CPU) | 2.2 ms |
| fetch errors | 0 |

The 10 misses were one run of consecutive frames, not scattered; every other
frame had a face. The scores sit just above the 0.9 threshold, so a small
change in pose or lighting can push a visible face below it. Try
`--score 0.7` if faces that are clearly in view get missed.
