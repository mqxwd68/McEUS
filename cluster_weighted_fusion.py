"""
External Pseudo-Label Generation Pipeline via Multi-Detector Prompting and SAM2 Fusion.

This script executes a zero-cost pseudo-label expansion for unannotated external datasets.
It utilizes an ensemble of YOLO detectors (cross-validation folds) to generate candidate
bounding boxes, applies confidence-weighted clustering fusion to resolve overlapping prompts,
and utilizes SAM2 to generate high-fidelity segmentation masks.
Outputs are exported as both normalized YOLO polygons and indexed Palette PNGs.
"""

import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple
from ultralytics import SAM, YOLO
from torchvision.ops import nms
import colorsys
from PIL import Image
from tqdm import tqdm

# ==========================================
# 0. Environment & Device Configuration
# ==========================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ==========================================
# 1. Constants & Global Configurations
# ==========================================
# Median global confidence thresholds across distinct categories
CONF_THRESH_MAP = {0: 0.37, 1: 0.36, 2: 0.33, 3: 0.49, 4: 0.42, 5: 0.48, 6: 0.56} #

# Semantic mask mapping: Index 0 strictly reserved for background.
COLOR_MAP = {
    0: [0, 0, 0],       # Background (Index 0)
    1: [255, 0, 0],     # Class 1    (YOLO id 0 -> Red)
    2: [0, 255, 0],     # Class 2    (YOLO id 1 -> Green)
    3: [0, 0, 255],     # Class 3    (YOLO id 2 -> Blue)
    4: [255, 255, 0],   # Class 4    (YOLO id 3 -> Yellow)
    5: [255, 0, 255],   # Class 5    (YOLO id 4 -> Magenta)
    6: [0, 255, 255],   # Class 6    (YOLO id 5 -> Cyan)
    7: [255, 255, 255]  # Class 7    (YOLO id 6 -> White)
}


def create_palette() -> list:
    """Generate a 256-color palette mapping with strict reservations for target classes."""
    palette = []
    for i in range(8):
        palette.extend(COLOR_MAP.get(i, [0, 0, 0]))

    # Populate remaining unused indices with procedural hues
    import colorsys
    for i in range(8, 256):
        hue = i * (360.0 / 248)
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        palette.extend([int(r * 255), int(g * 255), int(b * 255)])

    return palette


def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor,
                        iou_threshold: float = 0.6) -> torch.Tensor:
    """Class-independent NMS leveraging isolated coordinate offsets."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.int64)

    max_coordinate = boxes.max()
    offsets = classes.to(boxes.device) * (max_coordinate + 1000.0)
    boxes_for_nms = boxes + offsets[:, None]

    keep_indices = nms(boxes_for_nms, scores, iou_threshold)
    return keep_indices


def save_mask_as_png(mask: np.ndarray, output_path: Path, palette: list):
    """Export discrete label masks as embedded palette PNG files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask, mode='P')
    img.putpalette(palette)
    img.save(output_path)


# ==========================================
# 2. Core Geometric & Prompt Fusion Modules
# ==========================================
def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate standard Intersection over Union (IoU) between two bounding boxes."""
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area + 1e-6)


class BoxProcessor:
    @staticmethod
    def filter_by_confidence(boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Stage 1: Discard weak candidate proposals utilizing class-specific confidence thresholds."""
        keep_mask = np.zeros_like(confs, dtype=bool)
        for cls_id in np.unique(classes):
            cls_thresh = CONF_THRESH_MAP.get(int(cls_id), 0.25)
            cls_mask = (classes == cls_id)
            keep_mask[cls_mask] = confs[cls_mask] >= cls_thresh
        return boxes[keep_mask], confs[keep_mask], classes[keep_mask]

    @staticmethod
    def cluster_boxes(boxes: np.ndarray, confs: np.ndarray, iou_thresh: float) -> List[List[int]]:
        """Group bounding boxes based on spatial overlap (IoU) in descending confidence order."""
        if len(boxes) == 0:
            return []

        sorted_indices = np.argsort(-confs)
        clustered = []
        used = set()

        for idx in sorted_indices:
            if idx in used:
                continue

            current_cluster = [idx]
            used.add(idx)
            center_box = boxes[idx]

            for other_idx in sorted_indices:
                if other_idx not in used:
                    if calculate_iou(center_box, boxes[other_idx]) > iou_thresh:
                        current_cluster.append(other_idx)
                        used.add(other_idx)
            clustered.append(current_cluster)
        return clustered

    @staticmethod
    def apply_fusion(boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray, method: str, iou_thresh: float) -> \
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 2: Execute spatial aggregation strategy to synthesize final SAM2 prompts."""
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        if method == 'all_bboxes':
            return boxes, confs, classes

        elif method == 'nms':
            b_tensor = torch.tensor(boxes, device=device)
            s_tensor = torch.tensor(confs, device=device)
            c_tensor = torch.tensor(classes, device=device).float()

            keep_indices = non_max_suppression(b_tensor, s_tensor, c_tensor, iou_threshold=iou_thresh)
            if len(keep_indices) > 0:
                keep = keep_indices.cpu().numpy()
                return boxes[keep], confs[keep], classes[keep]
            return np.array([]), np.array([]), np.array([])

        fused_boxes, fused_classes, fused_confs = [], [], []

        for cls_id in np.unique(classes):
            cls_mask = classes == cls_id
            cls_boxes = boxes[cls_mask]
            cls_confs = confs[cls_mask]

            clusters = BoxProcessor.cluster_boxes(cls_boxes, cls_confs, iou_thresh)

            for cluster in clusters:
                cluster_boxes = cls_boxes[cluster]
                cluster_confs = cls_confs[cluster]

                if method == 'cluster_center':
                    fused_box = cluster_boxes[0]
                    conf = cluster_confs[0]
                elif method == 'intersection':
                    x_min = np.max(cluster_boxes[:, 0])
                    y_min = np.max(cluster_boxes[:, 1])
                    x_max = np.min(cluster_boxes[:, 2])
                    y_max = np.min(cluster_boxes[:, 3])
                    fused_box = cluster_boxes[0] if (x_min >= x_max or y_min >= y_max) else np.array(
                        [x_min, y_min, x_max, y_max])
                    conf = np.mean(cluster_confs)
                elif method == 'union':
                    x_min = np.min(cluster_boxes[:, 0])
                    y_min = np.min(cluster_boxes[:, 1])
                    x_max = np.max(cluster_boxes[:, 2])
                    y_max = np.max(cluster_boxes[:, 3])
                    fused_box = np.array([x_min, y_min, x_max, y_max])
                    conf = np.mean(cluster_confs)
                elif method == 'weighted_fusion':
                    weights = cluster_confs / cluster_confs.sum()
                    fused_box = np.sum(cluster_boxes * weights[:, None], axis=0)
                    conf = np.mean(cluster_confs)

                fused_boxes.append(fused_box)
                fused_confs.append(conf)
                fused_classes.append(cls_id)

        return np.array(fused_boxes), np.array(fused_confs), np.array(fused_classes)


class MaskPostProcessor:
    @staticmethod
    def apply_morphological_operations(mask: np.ndarray, kernel_size: int = 5, iters: int = 3) -> np.ndarray:
        """Apply structural opening and closing to bridge gaps and remove isolated speckles."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)
        return closed

    @staticmethod
    def apply_gaussian_smoothing(mask: np.ndarray, kernel_size: int = 15, n_iterations: int = 3) -> np.ndarray:
        """Apply iterative heavy Gaussian blurring combined with thresholding for sub-pixel edge curvature."""
        smoothed = mask.copy()
        for _ in range(n_iterations):
            smoothed = cv2.GaussianBlur(smoothed, (kernel_size, kernel_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        return smoothed

    @staticmethod
    def extract_yolo_polygons(mask: np.ndarray) -> List[List[float]]:
        """Extract continuous external boundaries and approximate into high-fidelity YOLO polygons."""
        img_h, img_w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                # Reduced epsilon scalar (0.0005) ensures dense vertex retention mirroring smooth curves
                epsilon = 0.0005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()

                if approx.ndim == 2 and len(approx) >= 3:
                    contour_norm = approx.astype(float)
                    contour_norm[:, 0] /= img_w
                    contour_norm[:, 1] /= img_h
                    polygons.append(contour_norm.flatten().tolist())
        return polygons


# ==========================================
# 3. Execution Pipeline
# ==========================================
def run_fusion_pipeline(args):
    print(f"Initializing Multi-Detector Pipeline with fusion strategy: [{args.method}]...")

    data_root = Path(args.data_root)
    sam_model = SAM(args.sam_model_path)

    # Pre-load all YOLO ensemble models to memory to optimize inference loops
    yolo_models = []
    for p in args.folds:
        model_path = Path(args.yolo_model_base_dir) / f"yolo_fold_{p}" / "best.pt"
        if model_path.exists():
            yolo_models.append(YOLO(str(model_path)))
        else:
            print(f"Warning: Model not found at {model_path}")

    if not yolo_models:
        raise FileNotFoundError("No valid YOLO models found. Check fold directories.")

    # Auto-discover all patient sub-directories in the external dataset root
    patients = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    global_palette = create_palette()

    for pid in patients:
        img_dir = data_root / pid / "images"
        if not img_dir.exists():
            continue

        print(f"\nProcessing External Cohort PID: {pid}")
        out_png_dir = data_root / pid / "masks_pseudo"
        out_yolo_dir = data_root / pid / "masks_pseudo_yolo"
        out_png_dir.mkdir(parents=True, exist_ok=True)
        out_yolo_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.jpg"))

        for img_path in tqdm(images, desc=f"Annotating {pid}"):
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue

            img_h, img_w = orig_img.shape[:2]
            img_prefix = img_path.stem

            # 1. Aggregate predictions from all active YOLO ensemble detectors
            raw_boxes, raw_confs, raw_classes = [], [], []
            for det_model in yolo_models:
                res = det_model.predict(source=orig_img, conf=0.01, iou=0.99, imgsz=args.imgsz, verbose=False)[0]
                if len(res.boxes.cls) > 0:
                    raw_boxes.append(res.boxes.xyxy.cpu().numpy())
                    raw_confs.append(res.boxes.conf.cpu().numpy())
                    raw_classes.append(res.boxes.cls.cpu().numpy().astype(int))

            if not raw_boxes:
                continue

            all_boxes = np.vstack(raw_boxes)
            all_confs = np.concatenate(raw_confs)
            all_classes = np.concatenate(raw_classes)

            # 2. Stage 1: Absolute Confidence Filtering
            f_boxes, f_confs, f_classes = BoxProcessor.filter_by_confidence(all_boxes, all_confs, all_classes)

            # 3. Stage 2: Prompt Fusion Execution
            prompt_boxes, prompt_confs, prompt_classes = BoxProcessor.apply_fusion(
                f_boxes, f_confs, f_classes, method=args.method, iou_thresh=args.iou_thresh
            )

            if len(prompt_boxes) == 0:
                continue

            # Strict image boundary clipping
            prompt_boxes[:, [0, 2]] = np.clip(prompt_boxes[:, [0, 2]], 0, img_w)
            prompt_boxes[:, [1, 3]] = np.clip(prompt_boxes[:, [1, 3]], 0, img_h)

            # 4. Stage 3: SAM2 Segmentation & Class Mask Aggregation
            processed_class_masks = {}
            for cls_id in np.unique(prompt_classes):
                cls_mask = prompt_classes == cls_id
                cls_boxes = prompt_boxes[cls_mask]

                boxes_tensor = torch.tensor(cls_boxes, device=device)
                sam_res = sam_model(orig_img, bboxes=boxes_tensor, verbose=False, save=False)[0]

                raw_masks = sam_res.masks.data.cpu().numpy()
                aggregated_mask = np.max(raw_masks, axis=0)

                if aggregated_mask.shape != (img_h, img_w):
                    aggregated_mask = cv2.resize(aggregated_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                aggregated_mask = (aggregated_mask > 0.5).astype(np.uint8) * 255

                # High-fidelity morphological rounding & smoothing
                processed = MaskPostProcessor.apply_morphological_operations(aggregated_mask, kernel_size=5, iters=3)
                processed = MaskPostProcessor.apply_gaussian_smoothing(processed, kernel_size=15, n_iterations=3)

                if np.any(processed):
                    processed_class_masks[cls_id] = processed

            # 5. Stage 4: Rasterization & Export
            mask_indexed = np.zeros((img_h, img_w), dtype=np.uint8)
            labels_content = []

            for cls_id, mask in processed_class_masks.items():
                polygons = MaskPostProcessor.extract_yolo_polygons(mask)
                for poly in polygons:
                    poly_str = " ".join(f"{v:.6f}" for v in poly)
                    labels_content.append(f"{cls_id} {poly_str}")

                # Align 0-indexed YOLO identities to 1-indexed Semantic Palettes
                mask_indexed[mask == 255] = int(cls_id) + 1

            if labels_content:
                with open(out_yolo_dir / f"{img_prefix}.txt", 'w') as f:
                    f.write("\n".join(labels_content))

            save_mask_as_png(mask_indexed, out_png_dir / f"{img_prefix}.png", global_palette)

    print("\n✅ External pseudo-label generation pipeline successfully completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Detector SAM2 Pseudo-Label Generation Pipeline.")

    # Core Algorithm Architecture
    parser.add_argument('--method', type=str, default='weighted_fusion',
                        choices=['nms', 'intersection', 'union', 'all_bboxes', 'cluster_center', 'weighted_fusion'],
                        help='Aggregation strategy for overlapping YOLO prompts.')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU clustering threshold.')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference dimensions for YOLO forward passes.')

    # Weights and Data Paths
    parser.add_argument('--data_root', type=str, default='Data_external', help='Root execution architecture.')
    parser.add_argument('--sam_model_path', type=str, default='weights/SAM_weights/sam2.1_t.pt',
                        help='SAM2 weights path.')
    parser.add_argument('--yolo_model_base_dir', type=str, default='weights/multi_detector', help='YOLO ensemble root.')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='YOLO cross-validation target folds.')

    args = parser.parse_args()
    run_fusion_pipeline(args)