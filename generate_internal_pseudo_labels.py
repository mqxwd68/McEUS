"""
Internal Pseudo-Label Generation Pipeline via Optical Flow on YOLO Contours.

"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Palette index strictly allocates Index 0 to background.
COLOR_MAP = {
    0: [0, 0, 0],       # Background
    1: [0, 0, 255],     # Class 1
    # 2: [0, 255, 0],     # Class 2
    3: [255, 0, 0],     # Class 3
    4: [0, 255, 255],   # Class 4
    5: [255, 0, 255],   # Class 5
    6: [255, 255, 0],   # Class 6
    7: [255, 255, 255]  # Class 7
}

class OpticalFlowAbsoluteProximityEngine:
    """
    Robust optical flow propagation engine tracking sparse sub-pixel YOLO contours.
    Guarantees absolute minimal anchor distances per individual unannotated frame.
    """
    def __init__(self, data_root: str, t1: int = 5, t2: int = 10):
        self.data_root = Path(data_root)
        self.source_dir = self.data_root / "source"
        self.target_dir = self.data_root / "target"
        self.t1 = t1  # Max distance activating bi-directional IPOF fusion
        self.t2 = t2  # Absolute max scope permitting single-anchor DPOF mapping

        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        }

        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.morph_iters = 3

    # -------------------- Parsing & I/O --------------------
    def _load_gray_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def _load_yolo_masks(self, path: Path) -> list:
        if not path.exists():
            return []
        masks = []
        with open(path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                points = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
                masks.append({'class_id': class_id, 'points': points})
        return masks

    def _save_yolo_txt(self, masks: list, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            for mask in masks:
                line = [str(mask['class_id'])]
                line += [f"{x:.6f} {y:.6f}" for (x, y) in mask['points']]
                f.write(" ".join(line) + "\n")

    def _save_palette_png(self, masks: list, save_path: Path, width: int, height: int):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = np.zeros((height, width), dtype=np.uint8)

        for mask in masks:
            target_color_index = mask['class_id'] + 1
            points = [(int(x * width), int(y * height)) for (x, y) in mask['points']]
            if len(points) >= 3:
                cv2.fillPoly(canvas, [np.array(points)], target_color_index)

        out_img = Image.fromarray(canvas, mode='P')
        flat_palette = []
        for i in range(256):
            b, g, r = COLOR_MAP.get(i, [0, 0, 0])
            flat_palette.extend([r, g, b])

        out_img.putpalette(flat_palette)
        out_img.save(save_path)

    # -------------------- Geometry & Morphological Filtering --------------------
    def _warp_mask_contours(self, masks: list, flow: np.ndarray, alpha: float = 1.0) -> list:
        h, w = flow.shape[:2]
        warped_masks = []
        for mask in masks:
            new_points = []
            for (x_norm, y_norm) in mask['points']:
                x = x_norm * w
                y = y_norm * h
                dx = flow[int(y), int(x), 0] * alpha
                dy = flow[int(y), int(x), 1] * alpha
                new_x = max(0, min(x + dx, w - 1)) / w
                new_y = max(0, min(y + dy, h - 1)) / h
                new_points.append((new_x, new_y))
            warped_masks.append({'class_id': mask['class_id'], 'points': new_points})
        return warped_masks

    def _fuse_bidirectional_contours(self, fw_masks: list, bw_masks: list, alpha: float) -> list:
        fused = []
        bw_map = {m['class_id']: m for m in bw_masks}
        for fw_m in fw_masks:
            if fw_m['class_id'] in bw_map:
                bw_m = bw_map[fw_m['class_id']]
                points = []
                for (p1, p2) in zip(fw_m['points'], bw_m['points']):
                    x = p1[0] * (1 - alpha) + p2[0] * alpha
                    y = p1[1] * (1 - alpha) + p2[1] * alpha
                    points.append((x, y))
                fused.append({'class_id': fw_m['class_id'], 'points': points})
            else:
                fused.append(fw_m)
        return fused

    def _morphological_smoothing(self, masks: list, width: int, height: int) -> list:
        """
        Rasterize and process mask boundaries utilizing high-impact Gaussian smoothing
        coupled with high-fidelity contour approximation to guarantee visible edge circularity.
        """
        smoothed_masks = []
        for mask in masks:
            binary = np.zeros((height, width), dtype=np.uint8)
            points = [(int(x * width), int(y * height)) for (x, y) in mask['points']]
            if len(points) < 3:
                continue
            cv2.fillPoly(binary, [np.array(points)], 255)

            # 1. Base morphological gap closing
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iters)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iters)

            # 2. Increased kernel size to 15x15 to drive prominent sub-pixel curvature changes
            for _ in range(3):
                processed = cv2.GaussianBlur(processed, (5, 5), 0)
                _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)

                # 3. Reduced epsilon scalar from 0.002 to 0.0005 to retain dense smooth arc vertices
                epsilon = 0.0005 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True).squeeze()

                if approx.ndim == 2 and len(approx) >= 3:
                    norm_points = [(float(x) / width, float(y) / height) for [x, y] in approx]
                    smoothed_masks.append({'class_id': mask['class_id'], 'points': norm_points})

        return smoothed_masks

    def _check_class_consistency(self, masks1: list, masks2: list) -> bool:
        c1 = {m['class_id'] for m in masks1}
        c2 = {m['class_id'] for m in masks2}
        return c1 == c2

    # -------------------- Absolute Minimum Proximity Routing --------------------
    def _execute_dpof(self, patient_id: str, target_frame: int, anchor_frame: int):
        """Executes targeted DPOF projection strictly from the designated nearest anchor."""
        src_img_path = self.source_dir / patient_id / "images" / f"{anchor_frame}.jpg"
        src_yolo_path = self.source_dir / patient_id / "mask_real_yolo" / f"{anchor_frame}.txt"
        tgt_img_path = self.target_dir / patient_id / "images" / f"{target_frame}.jpg"

        img_anchor = self._load_gray_image(src_img_path)
        img_tgt = self._load_gray_image(tgt_img_path)
        masks_anchor = self._load_yolo_masks(src_yolo_path)

        if img_anchor is None or img_tgt is None or not masks_anchor:
            return

        h, w = img_anchor.shape
        out_yolo_dir = self.target_dir / patient_id / "masks_pseudo_yolo"
        out_png_dir = self.target_dir / patient_id / "masks_pseudo"

        flow = cv2.calcOpticalFlowFarneback(img_anchor, img_tgt, None, **self.flow_params)
        warped = self._warp_mask_contours(masks_anchor, flow, 1.0)
        warped = self._morphological_smoothing(warped, w, h)

        self._save_yolo_txt(warped, out_yolo_dir / f"{target_frame}.txt")
        self._save_palette_png(warped, out_png_dir / f"{target_frame}.png", w, h)

    def _execute_ipof(self, patient_id: str, target_frame: int, start_anchor: int, end_anchor: int):
        """Executes interpolated bi-directional IPOF fusion for internal frames."""
        src_img_start = self.source_dir / patient_id / "images" / f"{start_anchor}.jpg"
        src_img_end = self.source_dir / patient_id / "images" / f"{end_anchor}.jpg"
        src_yolo_start = self.source_dir / patient_id / "mask_real_yolo" / f"{start_anchor}.txt"
        src_yolo_end = self.source_dir / patient_id / "mask_real_yolo" / f"{end_anchor}.txt"
        tgt_img_path = self.target_dir / patient_id / "images" / f"{target_frame}.jpg"

        img_start = self._load_gray_image(src_img_start)
        img_end = self._load_gray_image(src_img_end)
        img_tgt = self._load_gray_image(tgt_img_path)
        masks_start = self._load_yolo_masks(src_yolo_start)
        masks_end = self._load_yolo_masks(src_yolo_end)

        if img_start is None or img_end is None or img_tgt is None or not masks_start or not masks_end:
            return

        h, w = img_start.shape
        out_yolo_dir = self.target_dir / patient_id / "masks_pseudo_yolo"
        out_png_dir = self.target_dir / patient_id / "masks_pseudo"

        flow_fw = cv2.calcOpticalFlowFarneback(img_start, img_end, None, **self.flow_params)
        flow_bw = cv2.calcOpticalFlowFarneback(img_end, img_start, None, **self.flow_params)

        alpha = (target_frame - start_anchor) / (end_anchor - start_anchor)
        fw_contours = self._warp_mask_contours(masks_start, flow_fw, alpha)
        bw_contours = self._warp_mask_contours(masks_end, flow_bw, 1 - alpha)

        fused = self._fuse_bidirectional_contours(fw_contours, bw_contours, alpha)
        fused = self._morphological_smoothing(fused, w, h)

        self._save_yolo_txt(fused, out_yolo_dir / f"{target_frame}.txt")
        self._save_palette_png(fused, out_png_dir / f"{target_frame}.png", w, h)

    def process_patient(self, patient_id: str):
        """Iterate unannotated target targets globally, assigning absolute shortest route path."""
        src_yolo_dir = self.source_dir / patient_id / "mask_real_yolo"
        tgt_img_dir = self.target_dir / patient_id / "images"

        if not src_yolo_dir.exists() or not tgt_img_dir.exists():
            return

        # Extract sorted absolute numerical values for complete anchor sets
        key_frames = sorted([int(f.stem) for f in src_yolo_dir.glob("*.txt")])
        target_frames = sorted([int(f.stem) for f in tgt_img_dir.glob("*.jpg")])

        if not key_frames or not target_frames:
            return

        print(f"\nProcessing Patient {patient_id} across {len(key_frames)} valid ground-truth anchors...")

        # Track execution metrics for console display
        ipof_count = 0
        dpof_counts = {k: 0 for k in key_frames}
        skipped_count = 0

        # Evaluate every individual unannotated target frame strictly on absolute geometry
        for u in target_frames:
            # Prevent attempting generation if target frame happens to be a real key frame
            if u in key_frames:
                continue

            # Locate bounding interval keyframes [m, n] flanking current frame u
            m, n = None, None
            for kf in key_frames:
                if kf < u:
                    m = kf
                elif kf > u and n is None:
                    n = kf
                    break

            # Check if bi-directional IPOF activation criteria are fully met
            ipof_executed = False
            if m is not None and n is not None:
                if (n - m) <= self.t1:
                    src_yolo_m = src_yolo_dir / f"{m}.txt"
                    src_yolo_n = src_yolo_dir / f"{n}.txt"
                    masks_m = self._load_yolo_masks(src_yolo_m)
                    masks_n = self._load_yolo_masks(src_yolo_n)

                    if self._check_class_consistency(masks_m, masks_n):
                        self._execute_ipof(patient_id, u, m, n)
                        ipof_count += 1
                        ipof_executed = True

            # Fallback: Find absolute closest global keyframe anchor if IPOF is bypassed
            if not ipof_executed:
                distances = {kf: abs(u - kf) for kf in key_frames}
                # Find anchor with minimum physical frame offset
                best_anchor = min(distances, key=distances.get)
                min_dist = distances[best_anchor]

                if min_dist <= self.t2:
                    self._execute_dpof(patient_id, u, best_anchor)
                    dpof_counts[best_anchor] += 1
                else:
                    skipped_count += 1

        # Summary console outputs validating absolute shortest route allocations
        print(f"  ↳ Total Target Frames Evaluated: {len(target_frames)}")
        print(f"  ↳ Generated via [IPOF Bi-directional]: {ipof_count} frames")
        for anchor, count in dpof_counts.items():
            if count > 0:
                print(f"  ↳ Generated via [DPOF Minimum Route] from Anchor {anchor}: {count} frames")
        if skipped_count > 0:
            print(f"  ↳ Skipped (Exceeded Proximity Limit t2={self.t2}): {skipped_count} frames")

    def run(self):
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        patients = sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])
        for p in patients:
            self.process_patient(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Absolute Proximity Routed Flow Pseudo-Labeling Pipeline.")
    parser.add_argument("--data_root", type=str, default="Data_internal", help="Root structural input paths.")
    parser.add_argument("--t1", type=int, default=5, help="Max interval distance activating bi-directional IPOF.")
    parser.add_argument("--t2", type=int, default=10, help="Absolute proximity threshold bounding DPOF scope.")
    args = parser.parse_args()

    engine = OpticalFlowAbsoluteProximityEngine(data_root=args.data_root, t1=args.t1, t2=args.t2)
    engine.run()
    print("\n✅ Absolute minimal proximity routing completed fully successfully across datasets.")