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
| SFace (face recognition) | `face_recognition_sface_2021dec.onnx` | 38,696,353 B | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

`2026may` is the zoo's default YuNet for OpenCV 5. It is the `2023mar` model
re-exported with dynamic input dims, which OpenCV 5's ONNX Runtime engine needs
for arbitrary frame sizes.

### Run

```sh
CAM_IP=192.168.20.71 .venv/bin/python pc/capture_detect.py --seconds 30
.venv/bin/python pc/capture_detect.py --source stream --preview --score 0.7   # MJPEG stream, every frame
.venv/bin/python pc/capture_detect.py --preview --warmup 10 --seconds 30   # watch it live
.venv/bin/python pc/capture_detect.py --seconds 60 --save   # keep frames that contain a face
```

Every second it fetches `http://$CAM_IP/capture` (default `192.168.20.71`;
`--interval 0` polls back to back), or with `--source stream` it reads the
MJPEG stream at `http://$CAM_IP:81/stream` and processes every frame. It runs YuNet (score ≥ 0.9, as in the zoo demo; change with `--score`), and prints
the timestamp, `t` (seconds since the measured part started), the time spent
waiting for the frame, detection time, number of faces, the best score (0 if no
face), and each face's score and box. At the end it prints the detection rate, mean score, and mean
fetch and detection times.

`--preview` shows each frame in a window, scaled up 2x, with the face box, its
score, `t`, and the running count of frames with a face. Press `q` to stop; the
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

### Second test session (2026-10-01)

All runs: stream unless noted, CPU, nothing saved.

**Frame rate (20 s each):**

| Source | fps | mean wait per frame | mean YuNet |
|---|---:|---:|---:|
| `/capture`, back to back (`--interval 0`) | 3.05 | 325 ms | 2.4 ms |
| MJPEG stream (port 81) | 4.43 | 221 ms | 2.4 ms |

The stream is about 45% faster because there is no new HTTP request per frame.
A face was in view during part of both runs (the protocol asked for nobody in
view). That barely matters for fps: detection is under 1% of the time per
frame, so the camera link sets the frame rate.

**Head-angle test** (`--score 0.7`, 40 s in four 10 s blocks: facing / ~30° /
~60° / profile): 168/183 frames with a face (92%), 30 of 168 detections between
0.7 and 0.9. The per-block numbers did not follow the intended angles (the ~60°
and profile blocks scored highest, 100% detection, mean 0.92/0.91), so the head
angle was most likely not held per block. They are not reported as an angle
result. What the run does show: all 15 misses came in two ~1.5 s gaps, and
before each one the score slid from ~0.86 to 0.72–0.79. A 0.7 threshold keeps a
turning face a few frames longer than 0.9 does. To redo: hold each angle until
the window's `t` passes 10, 20 and 30.

**Walk-by test** (`--score 0.7`, 60 s, six intended passes at ~2 m): 104/273
frames with a face (38%), in 9 groups of detections (a new group after ≥1 s
without a face):

| # | t (s) | frames with face | max score | box at max (px) | largest box (px) |
|---:|---|---:|---:|---:|---:|
| 1 | 0.0–1.4 | 7 | 0.912 | 77×97 | 84×101 |
| 2 | 5.5–7.6 | 10 | 0.927 | 99×113 | 97×122 |
| 3 | 10.6–14.4 | 18 | 0.921 | 67×105 | 95×113 |
| 4 | 20.5–23.9 | 16 | 0.919 | 82×104 | 100×126 |
| 5 | 30.7–33.2 | 12 | 0.933 | 72×98 | 81×100 |
| 6 | 37.5–39.1 | 8 | 0.908 | 82×108 | 106×108 |
| 7 | 44.0–45.9 | 10 | 0.918 | 68×87 | 90×97 |
| 8 | 51.4–53.2 | 9 | 0.907 | 91×101 | 92×101 |
| 9 | 56.5–60.1 | 14 | 0.857 | 88×90 | 92×92 |

Recorded as: **9 detection groups**; the seven middle ones (2–8, the
candidate passes) had **8–18 frames each and max score 0.91–0.93**. Groups 1
and 9 at the start and end of the run had 7 and 14 frames, max 0.912 and 0.857.
Per-pass time matching was skipped. No face narrower than 47 px was detected;
most boxes were 70–100 px wide in the 240x240 frame, so the passes probably came
closer than 2 m.

## Face recognition (SFace)

### Same-person threshold

SFace compares two aligned faces by the **cosine similarity** of their 128-d
features (or their norm-L2 distance). The threshold used here comes from
OpenCV, not a guess:

> "two faces have same identity if the cosine distance is greater than or equal
> to 0.363, or the normL2 distance is less than or equal to 1.128."
> — OpenCV tutorial *DNN-based Face Detection And Recognition*
> (https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)

The same values appear in OpenCV's `samples/dnn/face_detect.py`
(`cosine_similarity_threshold = 0.363`, `l2_similarity_threshold = 1.128`) and
in opencv_zoo's `models/face_recognition_sface/sface.py` at the pinned commit
(`_threshold_cosine = 0.363`, `_threshold_norml2 = 1.128`). The code uses
**cosine ≥ 0.363**; the cosine it computes matches `FaceRecognizerSF.match(...,
FR_COSINE)` exactly.

### Enroll

```sh
.venv/bin/python pc/enroll.py --name juan --seconds 30 --preview
```

Reads the stream for 30 s. For every frame with exactly one face (frames with
several faces are skipped, so nobody else is enrolled by accident), it aligns
and embeds the face with SFace. It keeps an embedding only if its cosine
similarity to every embedding already kept is below 0.9, up to 20, which gives
a diverse set. Only the embeddings are written, to `data/gallery/NAME.npy`
(float32, shape [k, 128], gitignored); no images. An existing gallery is not
overwritten without `--replace`.

### Recognize

```sh
.venv/bin/python pc/capture_detect.py --source stream --recognize --preview --score 0.7
```

Each detected face is compared with every embedding of every person in
`data/gallery/`. It gets the name with the highest similarity, or `unknown`
(red box) if that similarity is below 0.363. The name and similarity are drawn
in the preview and logged per face (`id=... sim=...`).

### First recognition test (2026-10-01)

- **Enrollment** (30 s, facing then turning and moving): 135 frames, 49 with
  one face, 0 with several. Kept **20 embeddings**; the cap was reached after
  ~15 s, so later poses were not sampled. Cosine between kept embeddings when
  added: 0.46–0.88.
- **Recognition walk-by** (60 s, six passes, stream, YuNet score ≥ 0.7):
  150 frames with a face, **145 recognized as juan (97%)**. Similarity as juan
  0.366–0.903; the 5 `unknown` faces were at 0.228–0.352. **12 detection
  groups, every one with at least one frame recognized.**

Limits of this test: only one person was enrolled and only that person walked
past, so false matches (a stranger named juan) were **not** tested. Enrollment
and test were minutes apart, with the same lighting and clothes. The weakest
correct matches (0.37–0.42) are close to the 0.363 threshold. Test with other
people, and on another day, before relying on it.
