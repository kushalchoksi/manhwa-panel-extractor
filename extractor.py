import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.color import rgb2gray, label2rgb
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class MangaPanelSegmenter:
    """
    A class for segmenting manga/manhwa pages into individual artwork panels,
    excluding speech bubbles and text elements.
    """

    def __init__(self, min_panel_area_ratio=0.01, edge_detection_sigma=1.0,
                 low_threshold=0.1, high_threshold=0.2):
        """
        Initialize the segmenter with configurable parameters.

        Args:
            min_panel_area_ratio: Minimum panel area as a ratio of page area
            edge_detection_sigma: Sigma parameter for Canny edge detection
            low_threshold: Low threshold for Canny edge detection
            high_threshold: High threshold for Canny edge detection
        """
        self.min_panel_area_ratio = min_panel_area_ratio
        self.edge_detection_sigma = edge_detection_sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        # Initialize attributes to None
        self.original_image = None
        self.grayscale = None
        self.masked_grayscale = None
        self.segmentation = None
        self.labels = None
        self.panels = []
        self.extracted_panels = []
        self.ordered_panels = []
        self.height = 0
        self.width = 0


    def load_image(self, image_path):
        """Load the manga page image."""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Could not read image file: {image_path}")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.height, self.width = self.original_image.shape[:2]
            return self.original_image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def preprocess_for_edge_detection(self):
        """
        Preprocess the image to highlight panel borders and reduce noise from text.
        Apply techniques to ignore speech bubbles extending outside panels.
        """
        if self.original_image is None: return None
        # Convert to grayscale
        grayscale = rgb2gray(self.original_image)
        self.grayscale = grayscale

        # Create a binary threshold to identify potential speech bubbles (white areas)
        speech_bubble_mask = grayscale > 0.9  # White/near-white areas

        # Dilate the speech bubble mask to ensure we capture the entire bubbles
        speech_bubble_mask = dilation(speech_bubble_mask, np.ones((5, 5)))

        # Create an edge detection mask that excludes speech bubbles
        # edge_mask = ~speech_bubble_mask # This wasn't actually used, removing

        # Apply the mask to the grayscale image to suppress speech bubbles
        masked_grayscale = grayscale.copy()
        masked_grayscale[speech_bubble_mask] = 0.5  # Set speech bubble areas to mid-gray

        self.masked_grayscale = masked_grayscale
        return masked_grayscale

    def detect_edges(self):
        """
        Detect edges in the image using Canny edge detection.
        Focus on panel boundaries while ignoring speech bubbles.
        """
        masked_grayscale = self.preprocess_for_edge_detection()
        if masked_grayscale is None: return None

        # Apply Canny edge detection on the masked grayscale
        edges = canny(
            masked_grayscale,
            sigma=self.edge_detection_sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )

        # Thicken edges through dilation to ensure panel borders are continuous
        # Applying dilation twice for potentially thicker lines
        thick_edges = dilation(dilation(edges))

        # Fill holes to create solid regions
        self.segmentation = ndi.binary_fill_holes(thick_edges)

        return self.segmentation

    def identify_panels(self):
        """
        Identify individual artwork panels from the segmented image.
        Ignore speech bubbles and focus on the main panel boundaries.
        """
        if self.segmentation is None or self.grayscale is None: return []

        # Label each region
        labels = label(self.segmentation)
        self.labels = labels

        # Extract region properties
        regions = regionprops(labels)

        # Find panel regions (excluding speech bubbles)
        initial_panels = []

        # Create a binary mask for potential speech bubbles
        speech_bubble_mask = self.grayscale > 0.9  # White areas

        # --- Simplified Panel Identification ---
        # Instead of complex overlap and merging, treat large non-background regions as potential panels
        min_area = self.min_panel_area_ratio * self.height * self.width

        for region in regions:
            # Check if region is large enough (ignore tiny artifacts and background)
            if region.area < min_area or region.label == 0: # region.label==0 is background
                continue

            y_min, x_min, y_max, x_max = region.bbox

            # --- Refined Check: Avoid primarily white regions (likely background gaps or large bubbles) ---
            region_mask_in_bbox = (labels == region.label)[y_min:y_max, x_min:x_max]
            region_pixels_in_bbox = self.grayscale[y_min:y_max, x_min:x_max][region_mask_in_bbox]
            white_ratio_in_region = np.sum(region_pixels_in_bbox > 0.9) / region.area
            
            # Skip regions that are primarily white (adjust threshold as needed)
            if white_ratio_in_region > 0.85:
                 # print(f"Skipping region {region.label} due to high white ratio: {white_ratio_in_region:.2f}")
                 continue

            initial_panels.append(region.bbox)


        # --- Merge Overlapping Bounding Boxes ---
        # This step helps combine regions that might be fragmented parts of the same panel
        if not initial_panels:
            self.panels = []
            return []

        merged = True
        while merged:
            merged = False
            merged_panels = []
            used = [False] * len(initial_panels)

            for i in range(len(initial_panels)):
                if used[i]:
                    continue
                current_bbox = list(initial_panels[i]) # Make mutable
                used[i] = True

                for j in range(i + 1, len(initial_panels)):
                    if used[j]:
                        continue
                    # Check for significant overlap (IoU or simple overlap ratio)
                    if self._do_bboxes_overlap(current_bbox, initial_panels[j]):
                         # Merge overlapping boxes
                         current_bbox = self._merge_bboxes(current_bbox, initial_panels[j])
                         used[j] = True
                         merged = True # Indicate that a merge happened in this pass

                merged_panels.append(tuple(current_bbox)) # Add the potentially merged box

            initial_panels = merged_panels # Update list for next potential merge pass

        # --- Final Filtering (Optional: Remove panels fully contained within others) ---
        final_panels = []
        initial_panels.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True) # Sort by area desc

        for i, p1 in enumerate(initial_panels):
            is_contained = False
            for j, p2 in enumerate(initial_panels):
                if i != j and self._is_bbox_contained(p1, p2):
                    is_contained = True
                    break
            if not is_contained:
                 # Further check: ensure it's not too small after merging/filtering
                 y_min, x_min, y_max, x_max = p1
                 if (y_max - y_min) * (x_max - x_min) >= min_area:
                    final_panels.append(p1)


        # Assign to self.panels
        self.panels = final_panels
        # print(f"Identified {len(self.panels)} panels.")
        return self.panels


    def extract_artwork_panels(self):
        """
        Extract the artwork panels from the original image based on identified bounding boxes.
        """
        if self.original_image is None: return []
        self.extracted_panels = []

        for i, bbox in enumerate(self.panels):
            # Ensure bbox coordinates are valid
            y_min, x_min, y_max, x_max = bbox
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(self.height, y_max)
            x_max = min(self.width, x_max)

            # Check if coordinates are still valid after clipping
            if y_min >= y_max or x_min >= x_max:
                 # print(f"Skipping invalid bbox after clipping: {bbox}")
                 continue

            # Extract the panel
            panel = self.original_image[y_min:y_max, x_min:x_max].copy()

            # Store the panel and its bounding box
            self.extracted_panels.append({
                'bbox': bbox,
                'panel': panel,
                'panel_id': i + 1 # Temporary ID, will be updated after ordering
            })

        return self.extracted_panels

    def order_panels(self, reading_order='ltr'):
        """
        Order the panels according to typical reading order.
        Args:
            reading_order (str): 'ltr' (left-to-right) or 'rtl' (right-to-left).
                                 Default is 'ltr'.
        """
        if not self.extracted_panels:
            self.ordered_panels = []
            return []

        # Sort panels primarily by their vertical position (top-to-bottom).
        # For panels roughly on the same horizontal line, sort by horizontal position.
        # A tolerance is used to group panels that are vertically close.
        vertical_tolerance = (self.height * 0.05) # e.g., 5% of page height tolerance

        # Group panels by approximate vertical position
        rows = []
        sorted_by_y = sorted(self.extracted_panels, key=lambda p: p['bbox'][0]) # Sort by y_min

        if not sorted_by_y:
            self.ordered_panels = []
            return []

        current_row = [sorted_by_y[0]]
        last_y_min = sorted_by_y[0]['bbox'][0]

        for panel in sorted_by_y[1:]:
            y_min = panel['bbox'][0]
            # If the current panel's top is close to the last panel's top, consider it in the same row
            if abs(y_min - last_y_min) < vertical_tolerance:
                current_row.append(panel)
                # Update last_y_min to the average or min y of the current row for better grouping?
                # For simplicity, let's keep comparing with the first element's top in the row group
            else:
                # Start a new row
                rows.append(current_row)
                current_row = [panel]
                last_y_min = y_min

        rows.append(current_row) # Add the last row

        # Sort panels within each row based on reading order
        self.ordered_panels = []
        for row in rows:
            if reading_order == 'rtl':
                # Sort right-to-left based on the left edge (x_min)
                sorted_row = sorted(row, key=lambda p: p['bbox'][1], reverse=True)
            else: # Default ltr
                # Sort left-to-right based on the left edge (x_min)
                sorted_row = sorted(row, key=lambda p: p['bbox'][1])
            self.ordered_panels.extend(sorted_row)

        # Reassign panel IDs based on the final order
        for i, panel in enumerate(self.ordered_panels):
            panel['panel_id'] = i + 1

        return self.ordered_panels

    def save_panels(self, output_dir, base_filename, original_extension):
        """
        Save the extracted artwork panels as individual image files using
        the specified naming convention.

        Args:
            output_dir (str): Directory to save the panels.
            base_filename (str): The original image filename without extension (e.g., "page001").
            original_extension (str): The original image file extension (e.g., ".jpg").
        """
        # This function assumes output_dir already exists
        if not self.ordered_panels:
            print("No panels to save.")
            return 0

        saved_count = 0
        for panel_data in self.ordered_panels:
            panel = panel_data['panel']
            panel_id = panel_data['panel_id']

            # Construct the output filename: base_filename_panelID.original_extension
            # Use 2-digit padding for panel ID
            output_filename = f"{base_filename}_{panel_id:02d}{original_extension}"
            output_path = os.path.join(output_dir, output_filename)

            try:
                # Convert back to BGR for saving with OpenCV/PIL if needed
                if panel.ndim == 3 and panel.shape[2] == 3: # Check if it's an RGB image
                     panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                else:
                     panel_bgr = panel # Assume grayscale or already BGR

                # Save the panel using PIL to handle various formats potentially better
                panel_img = Image.fromarray(panel) # Use original RGB panel for PIL
                panel_img.save(output_path)
                saved_count += 1
            except Exception as e:
                print(f"Error saving panel {output_path}: {e}")

        return saved_count

    def visualize_results(self, show_original=True, show_edges=True, show_panels=True):
        """
        Visualize the extraction results, showing the journey from source image to final panels.
        Displays multiple stages of the process with clear explanations.
        """
        if self.original_image is None:
            print("No image loaded to visualize.")
            return

        num_stages = 0
        if show_original: num_stages += 1
        if show_edges and self.segmentation is not None: num_stages +=1
        # Add more stages if needed

        plt.figure(figsize=(15, 5 * num_stages)) # Adjust figure size

        plot_index = 1

        # --- Plot 1: Original Image ---
        if show_original:
            plt.subplot(num_stages, 3, plot_index)
            plt.imshow(self.original_image)
            plt.title('1. Original Image')
            plt.axis('off')
            plot_index += 1

        # --- Plot 2: Edge Detection / Segmentation ---
        if show_edges and self.segmentation is not None:
             plt.subplot(num_stages, 3, plot_index)
             plt.imshow(self.segmentation, cmap='gray')
             plt.title('2. Edge Detection & Segmentation')
             plt.axis('off')
             plot_index += 1


        # --- Plot 3: Identified Panel Boundaries ---
        if self.panels:
            panel_boundary_img = self.original_image.copy()
            for bbox in self.panels:
                 y_min, x_min, y_max, x_max = [int(c) for c in bbox] # Ensure int
                 cv2.rectangle(panel_boundary_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) # Red boxes
            plt.subplot(num_stages, 3, plot_index)
            plt.imshow(panel_boundary_img)
            plt.title('3. Identified Panel Boundaries')
            plt.axis('off')
            plot_index += 1


        # --- Plot 4: Final Panels with Order ---
        if self.ordered_panels:
            img_with_boxes = self.original_image.copy()
            for panel_data in self.ordered_panels:
                bbox = panel_data['bbox']
                panel_id = panel_data['panel_id']
                y_min, x_min, y_max, x_max = [int(c) for c in bbox] # Ensure int

                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green boxes

                # Add panel number (position inside the box)
                text_x = x_min + 10
                text_y = y_min + 30
                # Basic check to keep text within image bounds
                if text_y > y_max: text_y = y_min + 15
                if text_x > x_max - 20 : text_x = x_min + 5

                cv2.putText(img_with_boxes, str(panel_id), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Blue text

            plt.subplot(num_stages, 3, plot_index) # Or adjust layout as needed
            plt.imshow(img_with_boxes)
            plt.title('4. Final Ordered Panels')
            plt.axis('off')
            plot_index += 1


        plt.tight_layout()
        plt.suptitle('Manga Panel Segmentation Stages', fontsize=16, y=1.02) # Adjust title pos
        plt.show()


        # --- Figure 2: Display Extracted Panels ---
        if show_panels and self.ordered_panels:
            num_panels = len(self.ordered_panels)
            if num_panels > 0:
                max_cols = 4 # Increased columns for potentially more panels
                rows = int(np.ceil(num_panels / max_cols))
                cols = min(max_cols, num_panels)

                fig_panels, panel_axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 1)) # Adjust size
                # Handle case of single panel or single row/column
                if num_panels == 1:
                     panel_axes = np.array([panel_axes]) # Make iterable
                panel_axes = panel_axes.flatten() # Flatten for easy iteration

                for i, panel_data in enumerate(self.ordered_panels):
                    panel_axes[i].imshow(panel_data['panel'])
                    panel_axes[i].set_title(f'Panel {panel_data["panel_id"]}')
                    panel_axes[i].axis('off')

                # Hide any unused subplots
                for i in range(num_panels, len(panel_axes)):
                    panel_axes[i].axis('off')

                plt.tight_layout()
                plt.suptitle('Extracted Panels', fontsize=16, y=1.0) # Adjust title pos
                plt.show()


    def _do_bboxes_overlap(self, a, b, threshold=0.1):
        """Check if two bounding boxes overlap significantly."""
        y_min_a, x_min_a, y_max_a, x_max_a = a
        y_min_b, x_min_b, y_max_b, x_max_b = b

        # Calculate intersection coordinates
        intersect_ymin = max(y_min_a, y_min_b)
        intersect_xmin = max(x_min_a, x_min_b)
        intersect_ymax = min(y_max_a, y_max_b)
        intersect_xmax = min(x_max_a, x_max_b)

        # Calculate intersection area
        intersect_height = max(0, intersect_ymax - intersect_ymin)
        intersect_width = max(0, intersect_xmax - intersect_xmin)
        intersection_area = intersect_height * intersect_width

        if intersection_area == 0:
            return False

        # Calculate area of bounding boxes
        area_a = (y_max_a - y_min_a) * (x_max_a - x_min_a)
        area_b = (y_max_b - y_min_b) * (x_max_b - x_min_b)

        # Check if intersection area is a significant portion of either box
        if area_a > 0 and (intersection_area / area_a) > threshold:
            return True
        if area_b > 0 and (intersection_area / area_b) > threshold:
            return True

        return False


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


# %% Processing Function
def process_image(image_path, output_dir, visualize=True,
                  edge_detection_sigma=1.0, low_threshold=0.1, high_threshold=0.2,
                  min_panel_area=0.01, reading_order='ltr'):
    """
    Process a single manga/manhwa image and extract panels.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save extracted panels.
        visualize (bool): Whether to show visualization of results.
        edge_detection_sigma (float): Sigma parameter for Canny edge detection.
        low_threshold (float): Low threshold for Canny edge detection.
        high_threshold (float): High threshold for Canny edge detection.
        min_panel_area (float): Minimum panel area as a ratio of page area.
        reading_order (str): 'ltr' or 'rtl' for panel ordering.

    Returns:
        int: Number of panels extracted, or 0 if processing failed.
    """
    print(f"Processing: {image_path}")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract base filename and extension for saving panels later
    path_obj = Path(image_path)
    base_filename = path_obj.stem
    original_extension = path_obj.suffix # Includes the dot (e.g., ".jpg")

    # Initialize segmenter with parameters
    segmenter = MangaPanelSegmenter(
        min_panel_area_ratio=min_panel_area,
        edge_detection_sigma=edge_detection_sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )

    # Process the image
    if segmenter.load_image(image_path) is None:
        return 0 # Image loading failed
    if segmenter.detect_edges() is None:
        print(f" Edge detection failed for {image_path}")
        return 0
    segmenter.identify_panels()
    segmenter.extract_artwork_panels()
    segmenter.order_panels(reading_order=reading_order)

    # Save panels
    num_panels = segmenter.save_panels(output_dir, base_filename, original_extension)
    if num_panels > 0:
        print(f"-> Successfully extracted {num_panels} panels to {output_dir}/")
    else:
        print(f"-> No panels extracted or saved for {image_path}")

    # Visualize results if requested
    if visualize:
        print(f" Visualizing results for {base_filename}{original_extension}...")
        segmenter.visualize_results(show_panels=(num_panels > 0)) # Only show panels if any were extracted

    return num_panels

# %% Main Function
def main():
    """
    Main function to run the manga panel segmentation from command line.
    Handles both single image and directory processing.
    """
    parser = argparse.ArgumentParser(description='Manga/Manhwa Panel Segmentation Tool')
    parser.add_argument('input_path', type=str,
                        help='Path to the manga/manhwa page image OR a directory containing images.')
    parser.add_argument('--output', type=str, default=None, # Default is None, logic will handle it
                        help='Output directory base. If input is a file, panels go here.'
                             ' If input is a directory, a sibling directory named <input_dir>_output is created,'
                             ' and this argument is ignored.')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the segmentation results for each image.')
    parser.add_argument('--min-panel-area', type=float, default=0.01,
                        help='Minimum panel area as a ratio of total page area (e.g., 0.01 = 1%%). Default: 0.01')
    parser.add_argument('--edge-sigma', type=float, default=1.5, # Slightly higher default sigma might help
                        help='Sigma for Gaussian filter in Canny edge detection. Default: 1.5')
    parser.add_argument('--low-threshold', type=float, default=0.1,
                        help='Low threshold for Canny edge detection (as fraction of max intensity). Default: 0.1')
    parser.add_argument('--high-threshold', type=float, default=0.2,
                        help='High threshold for Canny edge detection (as fraction of max intensity). Default: 0.2')
    parser.add_argument('--reading-order', type=str, default='ltr', choices=['ltr', 'rtl'],
                        help='Reading order for panel numbering (ltr or rtl). Default: ltr')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_base = args.output

    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input_path}")
        return

    # --- Determine if input is directory or file ---
    if input_path.is_dir():
        # --- Directory Processing ---
        print(f"Processing directory: {input_path}")

        # Create the output directory path (sibling with _output suffix)
        output_dir_name = f"{input_path.name}_output"
        output_dir_path = input_path.parent / output_dir_name
        print(f"Output will be saved in: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True) # Create the main output dir for the batch

        # Find image files (adjust extensions as needed)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))

        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return

        total_panels_extracted = 0
        num_files_processed = 0

        for img_file_path in sorted(image_files): # Process in sorted order
             # Output for each image's panels goes directly into the derived output_dir_path
             try:
                 num_panels = process_image(
                    str(img_file_path),
                    output_dir=str(output_dir_path), # Panels saved directly here
                    visualize=args.visualize,
                    edge_detection_sigma=args.edge_sigma,
                    low_threshold=args.low_threshold,
                    high_threshold=args.high_threshold,
                    min_panel_area=args.min_panel_area,
                    reading_order=args.reading_order
                 )
                 total_panels_extracted += num_panels
                 num_files_processed += 1
             except Exception as e:
                 print(f"!! Failed to process {img_file_path}: {e}")
                 # Optionally add more robust error handling here (e.g., log errors)

        print("\n" + "="*30)
        print("Batch Processing Summary:")
        print(f" Processed {num_files_processed} image files from '{input_path}'.")
        print(f" Total panels extracted: {total_panels_extracted}")
        print(f" Output saved in: '{output_dir_path}'")
        print("="*30)

    elif input_path.is_file():
        # --- Single File Processing ---
        print(f"Processing single file: {input_path}")

        # Determine output directory for single file
        if output_base is None:
            # If --output is not specified, create output dir next to the file
            output_dir_path = input_path.parent / f"{input_path.stem}_output"
        else:
            output_dir_path = Path(output_base)

        print(f"Output will be saved in: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

        try:
            num_panels = process_image(
                str(input_path),
                output_dir=str(output_dir_path),
                visualize=args.visualize,
                edge_detection_sigma=args.edge_sigma,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
                min_panel_area=args.min_panel_area,
                reading_order=args.reading_order
            )
            print(f"\nFinished processing. Extracted {num_panels} panels.")
        except Exception as e:
            print(f"!! Failed to process {input_path}: {e}")

    else:
        print(f"Error: Input path is neither a file nor a directory: {args.input_path}")


if __name__ == "__main__":
    main()