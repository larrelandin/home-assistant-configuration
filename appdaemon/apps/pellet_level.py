import os
import time
from typing import Optional, List, Tuple

import appdaemon.plugins.hass.hassapi as hass
from PIL import Image, ImageOps, ImageFilter, ImageDraw


class PelletLevel(hass.Hass):
    """
    Builds a B/W mask (pellets bright -> white), detects fill via the first/top
    row that has >= min_white_px total white pixels for >= min_run_rows
    consecutive rows. Maps that fill fraction to sacks via piecewise-linear
    calibration. Also saves a debug snapshot (rotated + cropped only) with a
    red line drawn at the detected level row.
    """

    def initialize(self):
        # IO
        self.camera_entity: str    = self.args["camera_entity"]
        self.ha_snapshot_path: str = self.args["ha_snapshot_path"]   # /config/... (HA Core)
        self.raw_path: str         = self.args["raw_path"]           # /homeassistant/... (AppDaemon)

        # Optional outputs
        self.mask_output_path: Optional[str]   = self.args.get("mask_output_path")   # save binary mask (optional)
        self.debug_snapshot_path: Optional[str] = self.args.get("debug_snapshot_path")  # save cropped color image w/ red line
        self.debug_line_thickness: int         = int(self.args.get("debug_line_thickness", 2))

        # Sensor
        self.sensor_entity: str = self.args.get("sensor_entity", "sensor.pellet_level")

        # Geometry
        self.rotate_deg: int = int(self.args.get("rotate_deg", 0))
        self.crop_left  = self._opt_int(self.args.get("crop_left_px"))
        self.crop_top   = self._opt_int(self.args.get("crop_top_px"))
        self.crop_w     = self._opt_int(self.args.get("crop_width_px"))
        self.crop_h     = self._opt_int(self.args.get("crop_height_px"))

        # Masking
        self.threshold: Optional[int] = self._opt_int(self.args.get("threshold"))  # 0..255 or None for Otsu
        self.median_radius: int = int(self.args.get("median_radius", 1))           # small denoise
        self.close_size: int = int(self.args.get("close_size", 3))                 # binary closing (3/5/7). 0=off

        # Level-line by total white-pixel COUNT per row (your preference)
        self.min_run_px: int   = int(self.args.get("min_run_px", 25))  # min total white pixels per row
        self.min_run_rows: int = int(self.args.get("min_run_rows", 3))  # must hold for N consecutive rows

        # Calibration: fill fraction (0..1) -> sacks (0..12)
        self.cal_points: List[Tuple[float, float]] = self._parse_calibration(self.args.get("filter", []))
        if not self.cal_points:
            self.cal_points = [(0.0, 0.0), (1.0, 12.0)]

        # Timing
        self.interval_sec: int    = int(self.args.get("interval_seconds", 300))
        self.process_delay: float = float(self.args.get("process_delay_sec", 1.0))
        self.process_wait_sec: float = float(self.args.get("process_wait_sec", 10.0))

        for p in filter(None, [self.raw_path, self.mask_output_path, self.debug_snapshot_path]):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        self.run_in(self._wait_for_camera, 1)

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
            self.call_service("camera/snapshot",
                entity_id=self.camera_entity,
                filename=self.ha_snapshot_path)
            self.run_in(self._process_once, self.process_delay)
        except Exception as e:
            self.log(f"tick error: {e}", level="ERROR")

    def _process_once(self, *_):
        try:
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

            # Load original
            img_orig = Image.open(self.raw_path).convert("RGB")

            # Rotate
            if self.rotate_deg % 360 != 0:
                img_rot = img_orig.rotate(self.rotate_deg, expand=True)
            else:
                img_rot = img_orig

            # Crop ROI
            crop = self._crop_box()
            if crop is not None:
                l, t, r, b = self._clamp_box(img_rot.size, crop)
                img_roi = img_rot.crop((l, t, r, b))
            else:
                img_roi = img_rot

            # ---- Save DEBUG snapshot (cropped only, no color/contrast) with red level line later ----
            debug_img = img_roi.copy() if self.debug_snapshot_path else None

            # Build grayscale -> normalize -> optional denoise
            g = img_roi.convert("L")
            g = ImageOps.autocontrast(g, cutoff=1)
            if self.median_radius > 0:
                g = g.filter(ImageFilter.MedianFilter(size=max(3, self.median_radius * 2 + 1)))

            # Threshold to mask (pellets bright -> white)
            thr = self.threshold if self.threshold is not None else self._otsu_threshold(g)
            mask = g.point(lambda p, T=thr: 255 if p >= T else 0).convert("L")

            # Binary closing to fill black pinholes in white
            if self.close_size and self.close_size >= 3 and self.close_size % 2 == 1:
                mask = mask.filter(ImageFilter.MaxFilter(self.close_size))
                mask = mask.filter(ImageFilter.MinFilter(self.close_size))

            # Optional: save mask
            if self.mask_output_path:
                mask.save(self.mask_output_path)

            w, h = mask.size

            # Raw white % (unfiltered by design)
            hist = mask.histogram()
            white = hist[255]
            total = w * h
            raw_white_frac = (white / total) if total else 0.0
            raw_white_pct = round(raw_white_frac * 100.0, 1)

            # --- Level-line by TOTAL white-pixel COUNT per row ---
            y_level = self._find_first_row_with_count(mask, self.min_run_px, self.min_run_rows)

            # Fill fraction from line (fallback to raw area if not found)
            if y_level is None:
                fill_frac = raw_white_frac
            else:
                fill_frac = 1.0 - (y_level / float(h))
            fill_frac = max(0.0, min(1.0, fill_frac))

            # Map to sacks
            sacks = self._interp_piecewise(self.cal_points, fill_frac)
            sacks = max(0.0, min(12.0, sacks))
            sacks = round(sacks, 1)
            fill_pct = round((sacks / 12.0) * 100.0, 0)

            # ---- Draw red line on the DEBUG snapshot (if enabled and a level was found) ----
            if debug_img is not None:
                try:
                    if y_level is not None:
                        draw = ImageDraw.Draw(debug_img)
                        y = max(0, min(h - 1, int(y_level)))
                        thickness = max(1, int(self.debug_line_thickness))
                        # draw horizontal line across width
                        for dy in range(-(thickness // 2), thickness - (thickness // 2)):
                            y_draw = max(0, min(h - 1, y + dy))
                            draw.line([(0, y_draw), (w - 1, y_draw)], fill=(255, 0, 0))
                    # save debug snapshot
                    os.makedirs(os.path.dirname(self.debug_snapshot_path), exist_ok=True)
                    debug_img.save(self.debug_snapshot_path, quality=90)
                    try:
                        now = time.time()
                        os.utime(self.debug_snapshot_path, (now, now))
                    except Exception:
                        pass
                except Exception as e:
                    self.log(f"debug snapshot save error: {e}", level="WARNING")

            # Publish sensor
            self.set_state(
                self.sensor_entity,
                state=str(sacks),
                attributes={
                    "unit_of_measurement": "sacks",
                    "friendly_name": "Pellet Level (Sacks)",
                    "PercentageOfWhitePixels": raw_white_pct,
                    "SacksPercentage": fill_pct,
                    "threshold_used": thr,
                    "width": w,
                    "height": h,
                    "level_row": y_level if y_level is not None else -1,
                    "min_run_px": self.min_run_px,
                    "min_run_rows": self.min_run_rows,
                },
            )

            self.log(f"Sacks={sacks} | Sacks%={fill_pct} | raw%={raw_white_pct} | thr={thr} | y={y_level}")
        except Exception as e:
            self.log(f"process error: {e}", level="ERROR")

    # ---- level-line by count (preferred) ----
    def _find_first_row_with_count(self, mask: Image.Image, min_white_px: int, min_rows: int) -> Optional[int]:
        """Topmost y where total white pixels in a row >= min_white_px
        for >= min_rows consecutive rows. mask is L (0/255)."""
        w, h = mask.size
        pixels = mask.load()
        consec = 0
        for y in range(h):
            white_count = 0
            for x in range(w):
                if pixels[x, y] == 255:
                    white_count += 1
            if white_count >= min_white_px:
                consec += 1
                if consec >= min_rows:
                    return y - (min_rows - 1)
            else:
                consec = 0
        return None

    # ---- calibration & helpers ----
    def _parse_calibration(self, filt) -> List[Tuple[float, float]]:
        if not isinstance(filt, list):
            return []
        pts: List[Tuple[float, float]] = []
        for item in filt:
            if isinstance(item, dict) and "calibrate_linear" in item:
                lines = item["calibrate_linear"]
                if not isinstance(lines, list):
                    continue
                for line in lines:
                    if isinstance(line, str) and "->" in line:
                        a, b = line.split("->", 1)
                        try:
                            x = float(a.strip())
                            y = float(b.strip())
                            pts.append((x, y))
                        except Exception:
                            pass
                    elif isinstance(line, (list, tuple)) and len(line) == 2:
                        try:
                            x = float(line[0])
                            y = float(line[1])
                            pts.append((x, y))
                        except Exception:
                            pass
        pts = sorted({(round(x, 6), round(y, 6)) for x, y in pts}, key=lambda t: t[0])
        return pts if len(pts) >= 2 else []

    def _interp_piecewise(self, pts, x):
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x0 <= x <= x1:
                if x1 == x0:
                    return (y0 + y1) / 2.0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)   # <-- correct
        return pts[-1][1]


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
