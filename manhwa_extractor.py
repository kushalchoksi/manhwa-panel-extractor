import numpy as np
import cv2
import argparse
from pathlib import Path
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('manhwa_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class ManhwaProcessor:
    """
    A unified class for processing manhwa pages, extracting panels, and refining them
    to remove speech bubbles using HSV color filtering.
    """
    
    def __init__(self, 
                 min_panel_area_ratio=0.01, 
                 edge_detection_sigma=1.0,
                 low_threshold=0.1, 
                 high_threshold=0.2,
                 saturation_threshold=20,
                 value_range=(30, 225),
                 min_content_area=0.1,
                 padding=5,
                 reading_order='ltr'):
        """
        Initialize the ManhwaProcessor with parameters for both extraction and refinement.
        
        Args:
            min_panel_area_ratio (float): Minimum panel area as a ratio of page area
            edge_detection_sigma (float): Sigma for Canny edge detection
            low_threshold (float): Low threshold for Canny edge detection
            high_threshold (float): High threshold for Canny edge detection
            saturation_threshold (int): Minimum saturation for colored content
            value_range (tuple): Brightness range for content (min, max)
            min_content_area (float): Minimum area ratio for valid content
            padding (int): Padding around content boundaries
            reading_order (str): 'ltr' or 'rtl' for panel ordering
        """
        self.min_panel_area_ratio = min_panel_area_ratio
        self.edge_detection_sigma = edge_detection_sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.saturation_threshold = saturation_threshold
        self.value_min, self.value_max = value_range
        self.min_content_area = min_content_area
        self.padding = padding
        self.reading_order = reading_order
        
        # Initialize attributes
        self.original_image = None
        self.height = 0
        self.width = 0
        self.panels = []
        self.extracted_panels = []
        self.ordered_panels = []
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
    def load_image(self, image_path):
        """Load the manhwa page image."""
        try:
            self.original_image = cv2.imread(str(image_path))
            if self.original_image is None:
                raise ValueError(f"Could not read image file: {image_path}")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.height, self.width = self.original_image.shape[:2]
            logger.info(f"Loaded image: {image_path}, size: {self.original_image.shape}")
            return self.original_image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def detect_edges(self):
        """Detect panel edges using Canny edge detection, suppressing speech bubbles."""
        if self.original_image is None:
            return None
        grayscale = rgb2gray(self.original_image)
        speech_bubble_mask = grayscale > 0.9
        speech_bubble_mask = dilation(speech_bubble_mask, np.ones((5, 5)))
        masked_grayscale = grayscale.copy()
        masked_grayscale[speech_bubble_mask] = 0.5
        edges = canny(
            masked_grayscale,
            sigma=self.edge_detection_sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )
        thick_edges = dilation(dilation(edges))
        segmentation = ndi.binary_fill_holes(thick_edges)
        logger.debug("Completed edge detection")
        return segmentation

    def identify_panels(self, segmentation):
        """Identify individual panels from the segmented image."""
        if segmentation is None:
            return []
        labels = label(segmentation)
        regions = regionprops(labels)
        grayscale = rgb2gray(self.original_image)
        initial_panels = []
        min_area = self.min_panel_area_ratio * self.height * self.width

        for region in regions:
            if region.area < min_area or region.label == 0:
                continue
            y_min, x_min, y_max, x_max = region.bbox
            region_mask = (labels == region.label)[y_min:y_max, x_min:x_max]
            region_pixels = grayscale[y_min:y_max, x_min:x_max][region_mask]
            white_ratio = np.sum(region_pixels > 0.9) / region.area
            if white_ratio > 0.85:
                continue
            initial_panels.append(region.bbox)

        # Merge overlapping bounding boxes
        merged = True
        while merged:
            merged = False
            merged_panels = []
            used = [False] * len(initial_panels)
            for i in range(len(initial_panels)):
                if used[i]:
                    continue
                current_bbox = list(initial_panels[i])
                used[i] = True
                for j in range(i + 1, len(initial_panels)):
                    if used[j]:
                        continue
                    if self._do_bboxes_overlap(current_bbox, initial_panels[j]):
                        current_bbox = self._merge_bboxes(current_bbox, initial_panels[j])
                        used[j] = True
                        merged = True
                merged_panels.append(tuple(current_bbox))
            initial_panels = merged_panels

        # Filter contained panels
        final_panels = []
        initial_panels.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        for i, p1 in enumerate(initial_panels):
            is_contained = False
            for j, p2 in enumerate(initial_panels):
                if i != j and self._is_bbox_contained(p1, p2):
                    is_contained = True
                    break
            if not is_contained:
                y_min, x_min, y_max, x_max = p1
                if (y_max - y_min) * (x_max - x_min) >= min_area:
                    final_panels.append(p1)

        self.panels = final_panels
        logger.info(f"Identified {len(self.panels)} panels")
        return self.panels

    def extract_panels(self):
        """Extract panels from the original image."""
        if self.original_image is None:
            return []
        self.extracted_panels = []
        for i, bbox in enumerate(self.panels):
            y_min, x_min, y_max, x_max = [max(0, min(c, d)) for c, d in 
                zip(bbox, (self.height, self.width, self.height, self.width))]
            if y_min >= y_max or x_min >= x_max:
                logger.warning(f"Skipping invalid bbox: {bbox}")
                continue
            panel = self.original_image[y_min:y_max, x_min:x_max].copy()
            self.extracted_panels.append({
                'bbox': bbox,
                'panel': panel,
                'panel_id': i + 1
            })
        logger.debug(f"Extracted {len(self.extracted_panels)} panels")
        return self.extracted_panels

    def order_panels(self):
        """Order panels based on reading order."""
        if not self.extracted_panels:
            self.ordered_panels = []
            return []
        vertical_tolerance = self.height * 0.05
        rows = []
        sorted_by_y = sorted(self.extracted_panels, key=lambda p: p['bbox'][0])
        current_row = [sorted_by_y[0]]
        last_y_min = sorted_by_y[0]['bbox'][0]
        for panel in sorted_by_y[1:]:
            y_min = panel['bbox'][0]
            if abs(y_min - last_y_min) < vertical_tolerance:
                current_row.append(panel)
            else:
                rows.append(current_row)
                current_row = [panel]
                last_y_min = y_min
        rows.append(current_row)
        self.ordered_panels = []
        for row in rows:
            sorted_row = sorted(row, key=lambda p: p['bbox'][1], 
                              reverse=(self.reading_order == 'rtl'))
            self.ordered_panels.extend(sorted_row)
        for i, panel in enumerate(self.ordered_panels):
            panel['panel_id'] = i + 1
        logger.debug(f"Ordered {len(self.ordered_panels)} panels")
        return self.ordered_panels

    def refine_panel(self, panel_image):
        """Refine a panel by removing speech bubbles using HSV filtering."""
        try:
            hsv = cv2.cvtColor(panel_image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            saturation_mask = cv2.threshold(s, self.saturation_threshold, 255, cv2.THRESH_BINARY)[1]
            value_mask = cv2.inRange(v, self.value_min, self.value_max)
            color_mask = cv2.bitwise_and(saturation_mask, value_mask)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.morph_kernel)
            color_mask = cv2.dilate(color_mask, self.morph_kernel, iterations=2)
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = panel_image.shape[0] * panel_image.shape[1]
            min_area = img_area * self.min_content_area
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if not significant_contours:
                significant_contours = contours
            content_mask = np.zeros_like(color_mask)
            cv2.drawContours(content_mask, significant_contours, -1, 255, -1)
            x, y, w, h = cv2.boundingRect(content_mask)
            img_h, img_w = panel_image.shape[:2]
            x = max(0, x - self.padding)
            y = max(0, y - self.padding)
            w = min(img_w - x, w + (self.padding * 2))
            h = min(img_h - y, h + (self.padding * 2))
            refined_panel = panel_image[y:y+h, x:x+w].copy()
            logger.debug("Refined a panel")
            return refined_panel, color_mask, (x, y, w, h)
        except Exception as e:
            logger.error(f"Error refining panel: {e}")
            return panel_image, None, None

    def process_image(self, image_path, output_dir, extract_only=False, refine_only=False, visualize=False):
        """
        Process a single manhwa image through the full pipeline or selected stages.
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save outputs
            extract_only (bool): Only perform panel extraction and save extracted panels
            refine_only (bool): Only perform panel refinement
            visualize (bool): Visualize results
            
        Returns:
            tuple: (number of panels processed, list of processed panel paths)
        """
        path_obj = Path(image_path)
        base_filename = path_obj.stem
        extension = path_obj.suffix
        refined_dir = Path(output_dir) / "refined"
        panels_dir = Path(output_dir) / "panels"
        
        if extract_only:
            panels_dir.mkdir(parents=True, exist_ok=True)
        if not extract_only or refine_only:
            refined_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing image: {image_path}")
        
        if refine_only:
            # Process single panel image
            panel_image = self.load_image(image_path)
            if panel_image is None:
                return 0, []
            refined_panel, color_mask, content_box = self.refine_panel(panel_image)
            output_path = refined_dir / f"{base_filename}_refined{extension}"
            cv2.imwrite(str(output_path), cv2.cvtColor(refined_panel, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved refined panel to: {output_path}")
            if visualize:
                self._visualize_refinement(panel_image, color_mask, content_box, refined_panel)
            return 1, [str(output_path)]
        
        # Full pipeline or extraction only
        if self.load_image(image_path) is None:
            return 0, []
        segmentation = self.detect_edges()
        self.identify_panels(segmentation)
        self.extract_panels()
        self.order_panels()
        
        if not self.ordered_panels:
            logger.warning(f"No panels extracted from {image_path}")
            return 0, []
        
        processed_paths = []
        for panel_data in self.ordered_panels:
            panel = panel_data['panel']
            panel_id = panel_data['panel_id']
            
            if extract_only:
                # Save extracted panel
                panel_filename = f"{base_filename}_{panel_id:02d}{extension}"
                panel_path = panels_dir / panel_filename
                cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved extracted panel to: {panel_path}")
                processed_paths.append(str(panel_path))
            else:
                # Refine panel immediately and save result
                refined_panel, color_mask, content_box = self.refine_panel(panel)
                refined_filename = f"{base_filename}_{panel_id:02d}_refined{extension}"
                refined_path = refined_dir / refined_filename
                cv2.imwrite(str(refined_path), cv2.cvtColor(refined_panel, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved refined panel to: {refined_path}")
                processed_paths.append(str(refined_path))
                if visualize:
                    self._visualize_refinement(panel, color_mask, content_box, refined_panel, panel_id)
        
        if visualize and not refine_only:
            self._visualize_extraction()
        
        return len(self.ordered_panels), processed_paths

    def _visualize_extraction(self):
        """Visualize the panel extraction process."""
        if self.original_image is None:
            logger.warning("No image to visualize")
            return
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.imshow(self.original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        if self.panels:
            img_with_boxes = self.original_image.copy()
            for panel_data in self.ordered_panels:
                bbox = panel_data['bbox']
                panel_id = panel_data['panel_id']
                y_min, x_min, y_max, x_max = [int(c) for c in bbox]
                cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, str(panel_id), (x_min + 10, y_min + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            plt.subplot(222)
            plt.imshow(img_with_boxes)
            plt.title('Ordered Panels')
            plt.axis('off')
        
        if self.ordered_panels:
            num_panels = len(self.ordered_panels)
            cols = min(4, num_panels)
            rows = int(np.ceil(num_panels / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
            axes = np.array(axes).flatten()
            for i, panel_data in enumerate(self.ordered_panels):
                axes[i].imshow(panel_data['panel'])
                axes[i].set_title(f'Panel {panel_data["panel_id"]}')
                axes[i].axis('off')
            for i in range(num_panels, len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
        
        plt.show()

    def _visualize_refinement(self, original, color_mask, content_box, refined, panel_id=None):
        """Visualize the panel refinement process."""
        stages = [
            ("Original Panel", original),
            ("Color Mask", color_mask),
        ]
        if content_box:
            debug_img = original.copy()
            x, y, w, h = content_box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            stages.append(("Content Boundary", debug_img))
        stages.append(("Refined Panel", refined))
        
        fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 5))
        for i, (title, img) in enumerate(stages):
            ax = axes[i]
            if img is None:
                ax.text(0.5, 0.5, "Not Available", ha="center", va="center")
            elif len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2RGB))
            ax.set_title(f"{title} {'Panel ' + str(panel_id) if panel_id else ''}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def _do_bboxes_overlap(self, a, b, threshold=0.1):
        """Check if two bounding boxes overlap significantly."""
        y_min_a, x_min_a, y_max_a, x_max_a = a
        y_min_b, x_min_b, y_max_b, x_max_b = b
        intersect_ymin = max(y_min_a, y_min_b)
        intersect_xmin = max(x_min_a, x_min_b)
        intersect_ymax = min(y_max_a, y_max_b)
        intersect_xmax = min(x_max_a, x_max_b)
        intersect_height = max(0, intersect_ymax - intersect_ymin)
        intersect_width = max(0, intersect_xmax - intersect_xmin)
        intersection_area = intersect_height * intersect_width
        if intersection_area == 0:
            return False
        area_a = (y_max_a - y_min_a) * (x_max_a - x_min_a)
        area_b = (y_max_b - y_min_b) * (x_max_b - x_min_b)
        return (area_a > 0 and (intersection_area / area_a) > threshold) or \
               (area_b > 0 and (intersection_area / area_b) > threshold)

    def _merge_bboxes(self, a, b):
        """Merge two overlapping bounding boxes."""
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3])
        )

    def _is_bbox_contained(self, inner_bbox, outer_bbox):
        """Check if inner_bbox is fully contained within outer_bbox."""
        y_min_i, x_min_i, y_max_i, x_max_i = inner_bbox
        y_min_o, x_min_o, y_max_o, x_max_o = outer_bbox
        return (y_min_o <= y_min_i and x_min_o <= x_min_i and
                y_max_o >= y_max_i and x_max_o >= x_max_i)

def process_image_wrapper(args):
    """Wrapper for parallel processing of images."""
    image_path, output_dir, params, extract_only, refine_only, visualize = args
    processor = ManhwaProcessor(**params)
    try:
        num_panels, processed_paths = processor.process_image(
            image_path, output_dir,
            extract_only=extract_only,
            refine_only=refine_only,
            visualize=visualize
        )
        return num_panels, processed_paths
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return 0, []

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Manhwa Processing Pipeline')
    parser.add_argument('input_path', help='Path to input image or directory')
    parser.add_argument('--output', help='Output directory base')
    parser.add_argument('--extract-only', action='store_true', default=False, 
                        help='Only extract panels and save them')
    parser.add_argument('--refine-only', action='store_true', default=False, 
                        help='Only refine panels')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--min-panel-area', type=float, default=0.01,
                        help='Minimum panel area ratio')
    parser.add_argument('--edge-sigma', type=float, default=1.0,
                        help='Sigma for Canny edge detection')
    parser.add_argument('--low-threshold', type=float, default=0.1,
                        help='Low threshold for Canny edge detection')
    parser.add_argument('--high-threshold', type=float, default=0.2,
                        help='High threshold for Canny edge detection')
    parser.add_argument('--saturation', type=int, default=20,
                        help='Minimum saturation for colored content')
    parser.add_argument('--value-min', type=int, default=30,
                        help='Minimum brightness value')
    parser.add_argument('--value-max', type=int, default=225,
                        help='Maximum brightness value')
    parser.add_argument('--min-content-area', type=float, default=0.1,
                        help='Minimum content area ratio')
    parser.add_argument('--padding', type=int, default=5,
                        help='Padding around content boundaries')
    parser.add_argument('--reading-order', choices=['ltr', 'rtl'], default='ltr',
                        help='Panel reading order')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel processes')

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    # Prepare parameters for ManhwaProcessor initialization
    params = {
        'min_panel_area_ratio': args.min_panel_area,
        'edge_detection_sigma': args.edge_sigma,
        'low_threshold': args.low_threshold,
        'high_threshold': args.high_threshold,
        'saturation_threshold': args.saturation,
        'value_range': (args.value_min, args.value_max),
        'min_content_area': args.min_content_area,
        'padding': args.padding,
        'reading_order': args.reading_order
    }

    if input_path.is_dir():
        output_dir = input_path.parent / f"{input_path.name}_output" if args.output is None else Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"**/{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        total_panels = 0
        processed_paths = []
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            tasks = [(str(img), str(output_dir), params, args.extract_only, args.refine_only, args.visualize) 
                     for img in sorted(image_files)]
            results = list(tqdm(executor.map(process_image_wrapper, tasks), total=len(tasks)))
        
        for num_panels, paths in results:
            total_panels += num_panels
            processed_paths.extend(paths)
        
        logger.info(f"Processed {len(image_files)} images, extracted {total_panels} panels")
        logger.info(f"Output saved in: {output_dir}")

    else:
        output_dir = input_path.parent / f"{input_path.stem}_output" if args.output is None else Path(args.output)
        processor = ManhwaProcessor(**params)
        num_panels, processed_paths = processor.process_image(
            str(input_path), str(output_dir),
            extract_only=args.extract_only,
            refine_only=args.refine_only,
            visualize=args.visualize
        )
        logger.info(f"Processed {input_path}, extracted {num_panels} panels")
        logger.info(f"Output saved in: {output_dir}")

if __name__ == "__main__":
    main()