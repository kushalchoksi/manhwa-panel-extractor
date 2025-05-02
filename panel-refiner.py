import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PanelColorExtractor:
    """
    Extracts colored content from manga/manhwa panels by filtering out speech bubbles
    using HSV color space analysis and finding the actual content boundaries.
    """
    
    def __init__(self, saturation_threshold=20, value_range=(30, 225), 
                 min_content_area=0.1, padding=5):
        """
        Initialize the color extractor.
        
        Args:
            saturation_threshold (int): Minimum saturation to consider as colored content
            value_range (tuple): Range of brightness values to consider (min, max)
            min_content_area (float): Minimum area ratio to consider as valid content
            padding (int): Padding to add around detected content boundaries
        """
        self.saturation_threshold = saturation_threshold
        self.value_min = value_range[0]
        self.value_max = value_range[1]
        self.min_content_area = min_content_area
        self.padding = padding
        
        # Create kernels for morphological operations
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Store intermediate results
        self.original_panel = None
        self.hsv_image = None
        self.color_mask = None
        self.content_box = None
        self.refined_panel = None
    
    def load_panel(self, image_path):
        """Load the panel image."""
        try:
            self.original_panel = cv2.imread(str(image_path))
            if self.original_panel is None:
                raise ValueError(f"Could not read image file: {image_path}")
                
            # Convert to HSV for color analysis
            self.hsv_image = cv2.cvtColor(self.original_panel, cv2.COLOR_BGR2HSV)
            print(f"Loaded panel: {image_path}, size: {self.original_panel.shape}")
            return self.original_panel
        except Exception as e:
            print(f"Error loading panel: {e}")
            return None
    
    def extract_colored_content(self):
        """
        Extract colored content from the panel, excluding grayscale areas like speech bubbles.
        """
        if self.hsv_image is None:
            print("Error: Panel not loaded.")
            return None
        
        # Get HSV channels
        h, s, v = cv2.split(self.hsv_image)
        
        # 1. Create a mask for colored pixels (sufficient saturation)
        saturation_mask = cv2.threshold(s, self.saturation_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # 2. Create a mask for reasonable brightness range (exclude pure white/black)
        value_mask = cv2.inRange(v, self.value_min, self.value_max)
        
        # 3. Combine masks to get colored content
        self.color_mask = cv2.bitwise_and(saturation_mask, value_mask)
        
        # 4. Clean up the mask with morphological operations
        self.color_mask = cv2.morphologyEx(self.color_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        self.color_mask = cv2.morphologyEx(self.color_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # 5. Dilate to connect nearby regions
        self.color_mask = cv2.dilate(self.color_mask, self.morph_kernel, iterations=2)
        
        return self.color_mask
    
    def find_content_boundaries(self):
        """
        Find the bounding box of the colored content.
        """
        if self.color_mask is None:
            print("Error: Color mask not created.")
            return None
        
        # Find contours in the color mask
        contours, _ = cv2.findContours(
            self.color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            print("Warning: No colored content detected.")
            h, w = self.original_panel.shape[:2]
            self.content_box = (0, 0, w, h)
            return self.content_box
        
        # Filter significant contours (by area)
        img_area = self.original_panel.shape[0] * self.original_panel.shape[1]
        min_area = img_area * self.min_content_area
        
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not significant_contours:
            print("Warning: No significant colored regions found.")
            significant_contours = contours  # Use all contours if none are significant enough
        
        # Create a mask with all significant contours
        content_mask = np.zeros_like(self.color_mask)
        cv2.drawContours(content_mask, significant_contours, -1, 255, -1)
        
        # Find the bounding rectangle containing all colored content
        x, y, w, h = cv2.boundingRect(content_mask)
        
        # Add padding (ensuring we stay within image bounds)
        img_h, img_w = self.original_panel.shape[:2]
        
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(img_w - x, w + (self.padding * 2))
        h = min(img_h - y, h + (self.padding * 2))
        
        self.content_box = (x, y, w, h)
        return self.content_box
    
    def extract_refined_panel(self):
        """
        Extract the portion of the panel containing the colored content.
        """
        if self.content_box is None or self.original_panel is None:
            print("Error: Content boundaries not determined.")
            return None
        
        x, y, w, h = self.content_box
        
        # Create refined panel by cropping the original
        self.refined_panel = self.original_panel[y:y+h, x:x+w].copy()
        return self.refined_panel
    
    def process_panel(self, image_path, output_path=None, visualize=False):
        """
        Process a panel to extract colored content.
        
        Args:
            image_path: Path to the panel image
            output_path: Optional path to save the refined panel
            visualize: Whether to display visualization
            
        Returns:
            The refined panel image
        """
        # Load the panel
        if self.load_panel(image_path) is None:
            return None
        
        # Extract colored content
        self.extract_colored_content()
        
        # Find content boundaries
        self.find_content_boundaries()
        
        # Extract refined panel
        self.extract_refined_panel()
        
        # Save the result if path provided
        if output_path:
            output_path_obj = Path(output_path)
            
            # Ensure output directory exists
            if not output_path_obj.suffix:
                # Directory path
                output_dir = output_path_obj
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename based on input
                input_filename = Path(image_path).stem
                output_filename = f"{input_filename}_refined.png"
                output_path_full = output_dir / output_filename
            else:
                # File path with extension
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                output_path_full = output_path_obj
            
            # Save the refined panel
            cv2.imwrite(str(output_path_full), self.refined_panel)
            print(f"Saved refined panel to: {output_path_full}")
        
        # Show visualization if requested
        if visualize:
            self.visualize()
        
        return self.refined_panel
    
    def visualize(self):
        """Visualize the extraction process in a 3x3 grid."""
        if self.original_panel is None:
            print("No panel to visualize.")
            return

        # Prepare figure with a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        # 1. First row: Original Panel, HSV Image, Color Mask
        stages = [
            ("Original Panel", self.original_panel, cv2.COLOR_BGR2RGB),
            ("HSV Image", self.hsv_image, cv2.COLOR_HSV2RGB),
            ("Color Mask", self.color_mask, None)  # Grayscale
        ]

        for i, (title, img, color_conversion) in enumerate(stages):
            ax = axes[0, i]
            if img is None:
                ax.text(0.5, 0.5, "Not Available", ha="center", va="center")
            elif len(img.shape) == 2:  # Grayscale
                ax.imshow(img, cmap="gray")
            else:  # Color
                ax.imshow(cv2.cvtColor(img, color_conversion) if color_conversion else img)
            ax.set_title(title)
            ax.axis("off")

        # 2. Second row: Individual HSV Channels
        if self.hsv_image is not None:
            h, s, v = cv2.split(self.hsv_image)
            channels = [("Hue", h), ("Saturation", s), ("Value", v)]
            for i, (title, channel) in enumerate(channels):
                ax = axes[1, i]
                ax.imshow(channel, cmap="gray")
                ax.set_title(title)
                ax.axis("off")

        # 3. Third row: Significant Contours, Content Boundary, Refined Panel
        if self.color_mask is not None:
            ax = axes[2, 0]
            debug_img = self.original_panel.copy()
            contours, _ = cv2.findContours(self.color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = self.original_panel.shape[0] * self.original_panel.shape[1]
            min_area = img_area * self.min_content_area
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            cv2.drawContours(debug_img, significant_contours, -1, (0, 255, 0), 2)
            ax.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            ax.set_title("Significant Contours")
            ax.axis("off")

        if self.content_box is not None:
            ax = axes[2, 1]
            debug_img = self.original_panel.copy()
            x, y, w, h = self.content_box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ax.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            ax.set_title("Content Boundary")
            ax.axis("off")

        if self.refined_panel is not None:
            ax = axes[2, 2]
            ax.imshow(cv2.cvtColor(self.refined_panel, cv2.COLOR_BGR2RGB))
            ax.set_title("Refined Panel")
            ax.axis("off")

        # Adjust layout
        plt.tight_layout()
        plt.show()


def process_panel_with_color(image_path, output_path=None, visualize=False, 
                            saturation_threshold=20, value_range=(30, 225),
                            min_content_area=0.1, padding=5):
    """
    Process a manga/manhwa panel to extract colored content.
    
    Args:
        image_path (str): Path to the input panel image
        output_path (str, optional): Path to save the refined panel
        visualize (bool): Whether to show visualization
        saturation_threshold (int): Minimum saturation to consider as colored
        value_range (tuple): Range of brightness values to consider (min, max)
        min_content_area (float): Minimum area ratio to consider as valid content
        padding (int): Padding to add around detected content
        
    Returns:
        numpy.ndarray: The refined panel image
    """
    print(f"\n==== PROCESSING PANEL: {image_path} ====")
    
    # Create the color extractor
    extractor = PanelColorExtractor(
        saturation_threshold=saturation_threshold,
        value_range=value_range,
        min_content_area=min_content_area,
        padding=padding
    )
    
    # Process the panel
    try:
        refined_panel = extractor.process_panel(image_path, output_path, visualize)
        print(f"==== COMPLETED PROCESSING PANEL: {image_path} ====\n")
        return refined_panel
    except Exception as e:
        import traceback
        print(f"ERROR during processing: {e}")
        traceback.print_exc()
        return None


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manga Panel Color Extractor")
    parser.add_argument("input_path", help="Path to input panel image")
    parser.add_argument("--output", help="Path to save refined panel")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--saturation", type=int, default=20, 
                        help="Minimum saturation to consider as colored")
    parser.add_argument("--value-min", type=int, default=30,
                        help="Minimum brightness value")
    parser.add_argument("--value-max", type=int, default=225,
                        help="Maximum brightness value")
    parser.add_argument("--min-area", type=float, default=0.1,
                        help="Minimum content area ratio")
    parser.add_argument("--padding", type=int, default=5,
                        help="Padding around content boundaries")
    
    args = parser.parse_args()
    
    process_panel_with_color(
        args.input_path, 
        args.output,
        args.visualize,
        args.saturation,
        (args.value_min, args.value_max),
        args.min_area,
        args.padding
    )