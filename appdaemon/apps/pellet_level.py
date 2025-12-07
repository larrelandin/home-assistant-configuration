import os
import time
from typing import Optional, List, Tuple

import appdaemon.plugins.hass.hassapi as hass
from PIL import Image, ImageOps, ImageFilter


class PelletLevel(hass.Hass):
    """
    Snapshot -> rotate/crop -> grayscale normalize -> (denoise) -> threshold -> mask
    -> compute raw white% -> piecewise-linear calibration -> sacks + fill%

    Sensor state: number of sacks (0–12, one decimal)
    Attributes:
      - PercentageOfWhitePixels (raw % from mask, unfiltered)
      - FillLevelPercentage (calibrated %, 0–100)
    """

    # ---------- App setup ----------
    def initialize(self):
        # Required IO
        self.camera_entity: str    = self.args["camera_entity"]
        self.ha_snapshot_path: str = self.args["ha_snapshot_path"]   # /config/... (HA Core writes here)
        self.raw_path: str         = self.args["raw_path"]           # /homeassistant/... (AppDaemon reads)
        self.mask_output_path: Optional[str] = self.args.get("mask_output_path")  # optional visualization

        # Sensor settings
        self.sensor_entity: str = self.args.get("sensor_entity", "sensor.pellet_level")

        # Geometry
        self.rotate_deg: int = int(self.args.get("rotate_deg", 0))
        self.crop_left  = self._opt_int(self.args.get("crop_left_px"))
        self.crop_top   = self._opt_int(self.args.get("crop_top_px"))
        self.crop_w     = self._opt_int(self.args.get("crop_width_px"))
        self.crop_h     = self._opt_int(self.args.get("crop_height_px"))

        # Thresholding
        # If threshold is set (0..255) we use it; else Otsu auto
        self.threshold: Optional[int] = self._opt_int(self.args.get("threshold"))
        # Small denoise kills pepper specks (0=off, 1–2 typical)
        self.median_radius: int = int(self.args.get("median_radius", 1))

        # Calibration (piecewise linear). Maps raw fraction (0..1) -> sacks (0..12)
        self.cal_points: List[Tuple[float, float]] = self._parse_calibration(self.args.get("filter", []))
        if not self.cal_points:
            # Reasonable default: identity scaled to 12 sacks
            self.cal_points = [(0.0, 0.0), (1.0, 12.0)]

        # Timing
        self.interval_sec: int    = int(self.args.get("interval_seconds", 300))
        self.process_delay: float = float(self.args.get("process_delay_sec", 1.0))
        self.process_wait_sec: float = float(self.args.get("process_wait_sec", 10.0))

        # Ensure dirs (AppDaemon view)
        for p in filter(None, [self.raw_path, self.mask_output_path]):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        self.run_in(self._wait_for_camera, 1)

    # ---------- Loop ----------
    def _wait_for_camera(self, *_):
        if self.get_state(self.camera_entity) is None:
            self.log(f"Waiting for {self.camera_entity} to appear...", level="WARNING")
            self.run_in(self._wait_for_camera, 3)
            return
        self.log(f"Starting pellet level loop using {self.camera_entity}")
        self.run_in(self._tick, 1)
        self.run_every(self._tick, "now", self.interval_sec)

    def _tick(self, *_):
        try:
            # Ask HA Core to write snapshot (to /config/...)
            self.call_service("camera/snapshot", entity_id=self.camera_entity, filename=self.ha_snapshot_path)
            # Process shortly after
            self.run_in(self._process_once, self.process_delay)
        except Exception as e:
            self.log(f"tick error: {e}", level="ERROR")

    def _process_once(self, *_):
        try:
            # Wait briefly until raw exists & has size (AppDaemon path)
            deadline = time.time() + self.process_wait_sec
            while time.time() < deadline:
                try:
                    if os.path.exists(self.raw_path) and os.path.getsize(self.raw_path) > 0:
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            if not (os.path.exists(self.raw_path) and os.path.getsize(self.raw_path) > 0):
                self.log(f"Raw file not found (or empty) after wait: {self.raw_path}", level="WARNING")
                return

            # Load and ROI
            img = Image.open(self.raw_path).convert("RGB")
            if self.rotate_deg % 360 != 0:
                img = img.rotate(self.rotate_deg, expand=True)
            crop = self._crop_box()
            if crop is not None:
                l, t, r, b = self._clamp_box(img.size, crop)
                img = img.crop((l, t, r, b))

            # Grayscale + normalize + (optional) denoise
            g = img.convert("L")
            g = ImageOps.autocontrast(g, cutoff=1)
            if self.median_radius and self.median_radius > 0:
                g = g.filter(ImageFilter.MedianFilter(size=max(3, self.median_radius * 2 + 1)))

            # Threshold (pellets are brighter → white)
            thr = self.threshold if self.threshold is not None else self._otsu_threshold(g)
            mask = g.point(lambda p, T=thr: 255 if p >= T else 0).convert("L")  # keep 0/255 in L mode

            # Optional: save mask for inspection
            if self.mask_output_path:
                os.makedirs(os.path.dirname(self.mask_output_path), exist_ok=True)
                mask.save(self.mask_output_path)

            # Raw white percentage (unfiltered)
            hist = mask.histogram()  # 256 bins
            white = hist[255]
            total = mask.size[0] * mask.size[1]
            raw_frac = (white / total) if total else 0.0
            raw_pct = round(raw_frac * 100.0, 1)

            # Calibrate raw fraction -> sacks (piecewise linear)
            sacks = self._interp_piecewise(self.cal_points, raw_frac)
            sacks = max(0.0, min(12.0, sacks))
            sacks = round(sacks, 1)

            # Fill level percentage from sacks (0..12)
            fill_pct = round((sacks / 12.0) * 100.0, 1)

            # Publish main sensor (state = sacks)
            self.set_state(
                self.sensor_entity,
                state=str(sacks),
                attributes={
                    "unit_of_measurement": "sacks",
                    "friendly_name": "Pellet Level (Sacks)",
                    "PercentageOfWhitePixels": raw_pct,
                    "FillLevelPercentage": fill_pct,
                    "width": mask.size[0],
                    "height": mask.size[1],
                    "threshold_used": thr,
                },
            )

            self.log(f"Sacks={sacks} | raw%={raw_pct} | fill%={fill_pct} (thr={thr})")
        except Exception as e:
            self.log(f"process error: {e}", level="ERROR")

    # ---------- Calibration helpers ----------
    def _parse_calibration(self, filt) -> List[Tuple[float, float]]:
        """
        Accepts config like:
          filter:
            - calibrate_linear:
              - 0.02 -> 0
              - 0.28 -> 3.5
              ...
        Returns sorted list of (x, y) tuples with x in [0,1], y in sacks.
        """
        if not isinstance(filt, list):
            return []

        pairs: List[Tuple[float, float]] = []
        for item in filt:
            if not isinstance(item, dict):
                continue
            if "calibrate_linear" in item:
                lines = item["calibrate_linear"]
                if not isinstance(lines, list):
                    continue
                for line in lines:
                    if isinstance(line, str) and "->" in line:
                        a, b = line.split("->", 1)
                        try:
                            x = float(a.strip())
                            y = float(b.strip())
                            pairs.append((x, y))
                        except Exception:
                            pass
                    elif isinstance(line, (list, tuple)) and len(line) == 2:
                        try:
                            x = float(line[0])
                            y = float(line[1])
                            pairs.append((x, y))
                        except Exception:
                            pass
        # Deduplicate & sort by x
        pairs = sorted({(round(x, 6), round(y, 6)) for x, y in pairs}, key=lambda t: t[0])
        # Require at least two points
        return pairs if len(pairs) >= 2 else []

    def _interp_piecewise(self, pts: List[Tuple[float, float]], x: float) -> float:
        """Clamp + linear interpolate through (x,y) points."""
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        # find segment
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x0 <= x <= x1:
                # Avoid division by zero if duplicate x
                if x1 == x0:
                    return (y0 + y1) / 2.0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        # fallback (should not hit)
        return pts[-1][1]

    # ---------- Image helpers ----------
    def _opt_int(self, v):
        try:
            return None if v is None else int(v)
        except Exception:
            return None

    def _crop_box(self) -> Optional[tuple]:
        if None in (self.crop_left, self.crop_top, self.crop_w, self.crop_h):
            return None
        left = max(0, int(self.crop_left))
        top = max(0, int(self.crop_top))
        right = left + max(1, int(self.crop_w))
        bottom = top + max(1, int(self.crop_h))
        return (left, top, right, bottom)

    def _clamp_box(self, size, box):
        w, h = size
        l, t, r, b = box
        l = max(0, min(l, w))
        t = max(0, min(t, h))
        r = max(l, min(r, w))
        b = max(t, min(b, h))
        return (l, t, r, b)

    def _otsu_threshold(self, g_img: Image.Image) -> int:
        """Otsu threshold on 8-bit grayscale PIL image."""
        hist = g_img.histogram()
        total = sum(hist)
        sumB = wB = maximum = sum1 = 0
        for i in range(256):
            sum1 += i * hist[i]
        threshold = 127
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sum1 - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between >= maximum:
                threshold = i
                maximum = between
        return threshold
