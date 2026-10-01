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
