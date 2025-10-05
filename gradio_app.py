import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import zipfile
import tempfile
from PIL import Image

from manhwa_extractor import ManhwaProcessor

def process_manhwa_image(input_image, min_panel_area, edge_sigma, low_threshold,
                         high_threshold, reading_order, refine_panels):
    """
    Wrapper function to process manhwa image with Gradio inputs
    """
    if input_image is None:
        return None, "Please upload an image first", []

    try:
        img_array = np.array(input_image)

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save input image temporarily
            input_path = temp_dir_path / "input.png"
            cv2.imwrite(str(input_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            processor = ManhwaProcessor(
                min_panel_area_ratio=min_panel_area / 100,
                edge_detection_sigma=edge_sigma,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                reading_order=reading_order
            )

            processor.load_image(input_path)
            segmentation = processor.detect_edges()
            processor.identify_panels(segmentation)
            processor.extract_panels()
            processor.order_panels()

            if not processor.ordered_panels:
                return None, "No panels detected. Try adjusting the settings.", []

            panel_images = []
            panel_paths = []

            for panel_data in processor.ordered_panels:
                panel = panel_data['panel']
                panel_id = panel_data['panel_id']

                if refine_panels:
                    refined_panel, _, _ = processor.refine_panel(panel)
                    panel_to_save = refined_panel
                else:
                    panel_to_save = panel

                # Save panel temporarily for ZIP
                panel_filename = f"panel_{panel_id:02d}.png"
                panel_path = temp_dir_path / panel_filename
                cv2.imwrite(str(panel_path), cv2.cvtColor(panel_to_save, cv2.COLOR_RGB2BGR))
                panel_paths.append(panel_path)

                panel_images.append(Image.fromarray(panel_to_save))

            zip_path = temp_dir_path / "panels.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for panel_path in panel_paths:
                    zipf.write(panel_path, panel_path.name)

            persistent_zip = Path(tempfile.gettempdir()) / f"manhwa_panels_{id(input_image)}.zip"
            persistent_zip.write_bytes(zip_path.read_bytes())

            status_msg = f"Successfully extracted {len(processor.ordered_panels)} panels"
            if refine_panels:
                status_msg += " (with refinement)"

            return str(persistent_zip), status_msg, panel_images

    except Exception as e:
        return None, f"Error: {str(e)}", []

def create_demo():
    """Create the Gradio interface"""
    
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .settings-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .results-box {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
    }
    """
    
    with gr.Blocks(css=css, title="Manhwa Panel Extractor") as demo:

        gr.Markdown("# Manhwa Panel Extractor")

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## Upload Image")
                input_image = gr.Image(
                    label="Manhwa Page",
                    type="pil",
                    height=400
                )
                
                # Settings section
                with gr.Accordion("Advanced Settings", open=False):
                    gr.HTML('<div class="settings-box">')
                    
                    with gr.Row():
                        min_panel_area = gr.Slider(
                            minimum=0.1,
                            maximum=10.0,
                            value=1.0,
                            step=0.1,
                            label="Minimum Panel Area (%)",
                            info="Smaller values detect more panels"
                        )
                        
                        edge_sigma = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Edge Detection Sensitivity",
                            info="Higher values = smoother edges"
                        )
                    
                    with gr.Row():
                        low_threshold = gr.Slider(
                            minimum=0.01,
                            maximum=0.5,
                            value=0.1,
                            step=0.01,
                            label="Low Threshold",
                            info="Lower = more edge details"
                        )
                        
                        high_threshold = gr.Slider(
                            minimum=0.05,
                            maximum=0.8,
                            value=0.2,
                            step=0.01,
                            label="High Threshold",
                            info="Higher = stronger edges only"
                        )
                    
                    with gr.Row():
                        reading_order = gr.Radio(
                            choices=["ltr", "rtl"],
                            value="ltr",
                            label="Reading Order",
                            info="Left-to-right or Right-to-left"
                        )
                        
                        refine_panels = gr.Checkbox(
                            value=False,
                            label="Refine Panels",
                            info="Remove speech bubbles (experimental)"
                        )
                    
                    gr.HTML('</div>')
                
                # Process button
                process_btn = gr.Button(
                    "Extract Panels",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Results section
                gr.Markdown("## Results")
                
                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload an image and click 'Extract Panels' to begin..."
                )
                
                download_file = gr.File(
                    label="Download All Panels (ZIP)",
                    interactive=False
                )
                
                # Panel gallery
                panel_gallery = gr.Gallery(
                    label="Extracted Panels",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height=600,
                    object_fit="contain"
                )
        
        process_btn.click(
            fn=process_manhwa_image,
            inputs=[
                input_image,
                min_panel_area,
                edge_sigma,
                low_threshold,
                high_threshold,
                reading_order,
                refine_panels
            ],
            outputs=[download_file, status_message, panel_gallery]
        )
        
        # Example section
        gr.Markdown("""
        ## How to Use

        1. **Upload** your manhwa page image using the upload area
        2. **Adjust settings** if needed (optional - defaults work well for most images)
        3. **Click "Extract Panels"** to process the image
        4. **View results** in the gallery and download the ZIP file with all panels

        ### Tips
        - Higher **Edge Detection Sensitivity** works better for clean, high-contrast images
        - Lower **Minimum Panel Area** will detect smaller panels
        - Try **Refine Panels** to automatically remove speech bubbles
        - Use **Right-to-left** reading order for traditional manga layout
        """)

        # Examples (optional)
        with gr.Accordion("Example Images", open=False):
            gr.Markdown("*Upload your own manhwa images to test the extractor!*")
    
    return demo

# Additional utility functions for advanced features
def batch_process_images(image_files, settings):
    """
    Process multiple images at once
    """
    # Implementation for batch processing
    pass

def create_advanced_demo():
    """
    Create a more advanced demo with additional features
    """
    with gr.Blocks(title="Advanced Manhwa Panel Extractor") as demo:

        gr.Markdown("# Advanced Manhwa Panel Extractor")

        with gr.Tabs():
            # Single image processing tab
            with gr.TabItem("Single Image"):
                create_demo()

            # Batch processing tab
            with gr.TabItem("Batch Processing"):
                gr.Markdown("## Batch Process Multiple Images")

                batch_files = gr.File(
                    file_count="multiple",
                    label="Upload Multiple Images",
                    file_types=["image"]
                )

                batch_process_btn = gr.Button("Process All Images")
                batch_results = gr.File(label="Download Batch Results")

                # Add batch processing logic here

            # Settings presets tab
            with gr.TabItem("Presets"):
                gr.Markdown("## Settings Presets")
                
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "Default",
                        "High Detail",
                        "Large Panels Only",
                        "Webtoon Style",
                        "Traditional Manga"
                    ],
                    label="Choose Preset"
                )
                
                # Add preset logic here
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    
    # Launch options
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed error messages
    )