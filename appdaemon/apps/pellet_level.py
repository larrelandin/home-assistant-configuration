import os
import time
from typing import Optional, List, Tuple

import appdaemon.plugins.hass.hassapi as hass
from PIL import Image, ImageOps, ImageFilter, ImageDraw


class PelletLevel(hass.Hass):
    """
    Pipeline:
      1) snapshot -> rotate -> crop ROI
      2) grayscale -> FIXED mapping [fixed_black_level..fixed_white_level] -> optional gamma
      3) fixed threshold (or optional otsu/percentile) -> binary mask (pellets = white)
      4) morphology (optional opening/closing)
      5) find first/top row having enough white pixels for N consecutive rows
      6) map level fraction -> sacks via piecewise-linear calibration
      7) publish sensor + save debug (cropped color) with red level line + mask (optional)
    """

    # ---------------- init ----------------
    def initialize(self):
        # IO
        self.camera_entity: str = self.args["camera_entity"]
        self.ha_snapshot_path: str = self.args["ha_snapshot_path"]      # /config/... (HA Core writes)
        self.raw_path: str = self.args["raw_path"]                      # /homeassistant/... (AppDaemon reads)

        # Optional outputs
        self.mask_output_path: Optional[str] = self.args.get("mask_output_path")
        self.debug_snapshot_path: Optional[str] = self.args.get("debug_snapshot_path")
        self.debug_line_thickness: int = int(self.args.get("debug_line_thickness", 2))

        # Sensor
        self.sensor_entity: str = self.args.get("sensor_entity", "sensor.pellet_level")

        # Geometry
        self.rotate_deg: int = int(self.args.get("rotate_deg", 0))
        self.crop_left = self._opt_int(self.args.get("crop_left_px"))
        self.crop_top = self._opt_int(self.args.get("crop_top_px"))
        self.crop_w = self._opt_int(self.args.get("crop_width_px"))
        self.crop_h = self._opt_int(self.args.get("crop_height_px"))

        # Fixed mapping + thresholding (your request)
        self.threshold_mode: str = str(self.args.get("threshold_mode", "fixed")).lower()  # fixed|otsu|percentile
        self.threshold: Optional[int] = self._opt_int(self.args.get("threshold"))          # used if mode=fixed
        self.threshold_percentile: float = float(self.args.get("threshold_percentile", 90))

        self.fixed_black_level: int = int(self.args.get("fixed_black_level", 60))
        self.fixed_white_level: int = int(self.args.get("fixed_white_level", 180))
        self.gamma: float = float(self.args.get("gamma", 1.0))

        # Denoise & morphology
        self.median_radius: int = int(self.args.get("median_radius", 1))   # 0 = off
        self.open_size: int = int(self.args.get("open_size", 0))           # 0 = off, else 3/5/7
        self.close_size: int = int(self.args.get("close_size", 3))         # 0 = off, else 3/5/7

        # Level-line by total white-pixel count per row
        self.min_run_px: int = int(self.args.get("min_run_px", 25))        # absolute px requirement
        self.min_row_white_frac = self._opt_float(self.args.get("min_row_white_frac"))  # overrides px if set
        self.min_run_rows: int = int(self.args.get("min_run_rows", 3))      # consecutive rows needed
        self.scan_top_ignore_px: int = int(self.args.get("scan_top_ignore_px", 0))
        self.scan_bottom_ignore_px: int = int(self.args.get("scan_bottom_ignore_px", 0))
        self.no_line_fill_floor: float = float(self.args.get("no_line_fill_floor", 0.0))  # 0..1 if no line

        # Calibration: fraction (0..1) -> sacks (0..12)
        self.cal_points: List[Tuple[float, float]] = self._parse_calibration(self.args.get("filter", []))
        if not self.cal_points:
            self.cal_points = [(0.0, 0.0), (1.0, 12.0)]

        # Timing
        self.interval_sec: int = int(self.args.get("interval_seconds", 300))
        self.process_delay: float = float(self.args.get("process_delay_sec", 1.0))
        self.process_wait_sec: float = float(self.args.get("process_wait_sec", 10.0))

        for p in filter(None, [self.raw_path, self.mask_output_path, self.debug_snapshot_path]):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        self.run_in(self._wait_for_camera, 1)

    # ---------------- scheduling ----------------
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
            self.call_service(
                "camera/snapshot",
                entity_id=self.camera_entity,
                filename=self.ha_snapshot_path,
            )
            self.run_in(self._process_once, self.process_delay)
        except Exception as e:
            self.log(f"tick error: {e}", level="ERROR")

    # ---------------- core ----------------
    def _process_once(self, *_):
        try:
            # wait for HA to finish writing the raw file
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

            # 1) load -> rotate -> crop
            img = Image.open(self.raw_path).convert("RGB")
            if self.rotate_deg % 360 != 0:
                img = img.rotate(self.rotate_deg, expand=True)
            crop = self._crop_box()
            if crop is not None:
                l, t, r, b = self._clamp_box(img.size, crop)
                img_roi = img.crop((l, t, r, b))
            else:
                img_roi = img

            # Debug (cropped color) saved later with red line
            debug_img = img_roi.copy() if self.debug_snapshot_path else None

            # 2) grayscale + FIXED mapping + optional gamma (no auto ops)
            g = img_roi.convert("L")
            blk = max(0, min(255, int(self.fixed_black_level)))
            wht = max(0, min(255, int(self.fixed_white_level)))
            if wht <= blk:
                wht = blk + 1
            span = wht - blk

            # map [blk..wht] -> [0..255], clamp outside
            g_fixed = g.point(lambda v, b=blk, s=span: 0 if v <= b else (255 if v >= b + s else int((v - b) * 255.0 / s)))

            # optional gamma
            gam = float(self.gamma)
            if abs(gam - 1.0) > 1e-3:
                inv = 1.0 / max(1e-6, gam)
                lut = [int((i / 255.0) ** inv * 255.0 + 0.5) for i in range(256)]
                g_fixed = g_fixed.point(lut)

            # optional small denoise
            if self.median_radius > 0:
                g_fixed = g_fixed.filter(ImageFilter.MedianFilter(size=max(3, self.median_radius * 2 + 1)))

            # 3) threshold selection
            mode = self.threshold_mode
            if mode == "fixed" and self.threshold is not None:
                thr = int(self.threshold)
            elif mode == "percentile":
                hist = g_fixed.histogram()
                total = sum(hist)
                target = total * (float(self.threshold_percentile) / 100.0)
                cum = 0
                thr = 127
                for i, c in enumerate(hist):
                    cum += c
                    if cum >= target:
                        thr = i
                        break
            else:  # otsu
                thr = self._otsu_threshold(g_fixed)

            mask = g_fixed.point(lambda v, T=thr: 255 if v >= T else 0).convert("L")

            # 4) morphology
            if self.open_size and self.open_size >= 3 and self.open_size % 2 == 1:
                mask = mask.filter(ImageFilter.MinFilter(self.open_size))  # erode
                mask = mask.filter(ImageFilter.MaxFilter(self.open_size))  # dilate
            if self.close_size and self.close_size >= 3 and self.close_size % 2 == 1:
                mask = mask.filter(ImageFilter.MaxFilter(self.close_size))  # dilate
                mask = mask.filter(ImageFilter.MinFilter(self.close_size))  # erode

            # optional save
            if self.mask_output_path:
                try:
                    os.makedirs(os.path.dirname(self.mask_output_path), exist_ok=True)
                    mask.save(self.mask_output_path)
                except Exception as e:
                    self.log(f"mask save error: {e}", level="WARNING")

            w, h = mask.size

            # raw white area (debug only)
            hist = mask.histogram()
            white = hist[255]
            total = w * h
            raw_white_frac = (white / total) if total else 0.0
            raw_white_pct = round(raw_white_frac * 100.0, 1)

            # 5) level-line search (with top/bottom ignore + optional fractional target)
            y_level = self._find_first_row_with_count(mask, self.min_run_px, self.min_run_rows)

            if y_level is None:
                fill_frac = max(0.0, min(1.0, float(self.no_line_fill_floor)))
            else:
                fill_frac = 1.0 - (y_level / float(h))
            fill_frac = max(0.0, min(1.0, fill_frac))

            # 6) calibration -> sacks
            sacks = self._interp_piecewise(self.cal_points, fill_frac)
            sacks = max(0.0, min(12.0, sacks))
            sacks = round(sacks, 1)
            sacks_pct = round((sacks / 12.0) * 100.0, 0)

            # 7) debug line + save
            if debug_img is not None:
                try:
                    draw = ImageDraw.Draw(debug_img)
                    if y_level is not None:
                        y = max(0, min(h - 1, int(y_level)))
                        thickness = max(1, int(self.debug_line_thickness))
                        for dy in range(-(thickness // 2), thickness - (thickness // 2)):
                            y_draw = max(0, min(h - 1, y + dy))
                            draw.line([(0, y_draw), (w - 1, y_draw)], fill=(255, 0, 0))
                    os.makedirs(os.path.dirname(self.debug_snapshot_path), exist_ok=True)
                    debug_img.save(self.debug_snapshot_path, quality=90)
                    try:
                        now = time.time()
                        os.utime(self.debug_snapshot_path, (now, now))
                    except Exception:
                        pass
                except Exception as e:
                    self.log(f"debug snapshot save error: {e}", level="WARNING")

            # publish
            self.set_state(
                self.sensor_entity,
                state=str(sacks),
                attributes={
                    "unit_of_measurement": "sacks",
                    "friendly_name": "Pellet Level (Sacks)",
                    "PercentageOfWhitePixels": raw_white_pct,  # unfiltered area %
                    "FillLevelPercentage": sacks_pct,          # calibrated %
                    "LevelLineFraction": round(fill_frac, 3),  # 0..1
                    "threshold_used": thr,
                    "width": w,
                    "height": h,
                    "level_row": y_level if y_level is not None else -1,
                    "min_run_px": self.min_run_px,
                    "min_run_rows": self.min_run_rows,
                },
            )

            self.log(f"Sacks={sacks} | %={sacks_pct} | area%={raw_white_pct} | thr={thr} | y={y_level}")
        except Exception as e:
            self.log(f"process error: {e}", level="ERROR")

    # ---------------- detectors & helpers ----------------
    def _find_first_row_with_count(self, mask: Image.Image, min_white_px: int, min_rows: int) -> Optional[int]:
        """Topmost y where total white pixels in a row >= target for >= min_rows consecutive rows."""
        w, h = mask.size
        top = max(0, self.scan_top_ignore_px)
        bottom = h - max(0, self.scan_bottom_ignore_px)
        bottom = max(top, bottom)

        pixels = mask.load()
        consec = 0
        for y in range(top, bottom):
            white_count = 0
            for x in range(w):
                if pixels[x, y] == 255:
                    white_count += 1

            target = int(w * self.min_row_white_frac) if self.min_row_white_frac is not None else min_white_px

            if white_count >= target:
                consec += 1
                if consec >= min_rows:
                    return y - (min_rows - 1)
            else:
                consec = 0
        return None

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
                            pts.append((float(a.strip()), float(b.strip())))
                        except Exception:
                            pass
                    elif isinstance(line, (list, tuple)) and len(line) == 2:
                        try:
                            pts.append((float(line[0]), float(line[1])))
                        except Exception:
                            pass
        pts = sorted({(round(x, 6), round(y, 6)) for x, y in pts}, key=lambda t: t[0])
        return pts if len(pts) >= 2 else []

    def _interp_piecewise(self, pts: List[Tuple[float, float]], x: float) -> float:
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
                return y0 + t * (y1 - y0)
        return pts[-1][1]

    def _opt_int(self, v):
        try:
            return None if v is None else int(v)
        except Exception:
            return None

    def _opt_float(self, v):
        try:
            return None if v is None else float(v)
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
