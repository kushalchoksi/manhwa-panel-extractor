import gradio as gr
import numpy as np

from manhwa_extractor import ManhwaProcessor  # Uncomment when you have the module

def create_demo():
    """Create the Gradio interface"""
    
    # Custom CSS for styling
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
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Manhwa Panel Extractor</h1>
            <p>Upload your manhwa pages and extract individual panels automatically</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📤 Upload Image")
                input_image = gr.Image(
                    label="Manhwa Page",
                    type="pil",
                    height=400
                )
                
                # Settings section
                with gr.Accordion("⚙️ Advanced Settings", open=False):
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
                    "🚀 Extract Panels",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Results section
                gr.Markdown("## 🎯 Results")
                
                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload an image and click 'Extract Panels' to begin..."
                )
                
                download_file = gr.File(
                    label="📥 Download All Panels (ZIP)",
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
        
        # Connect the processing function
        process_btn.click(
            fn=ManhwaProcessor.extract_panels,
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
        ## 📖 How to Use
        
        1. **Upload** your manhwa page image using the upload area
        2. **Adjust settings** if needed (optional - defaults work well for most images)
        3. **Click "Extract Panels"** to process the image
        4. **View results** in the gallery and download the ZIP file with all panels
        
        ### 💡 Tips
        - Higher **Edge Detection Sensitivity** works better for clean, high-contrast images
        - Lower **Minimum Panel Area** will detect smaller panels
        - Try **Refine Panels** to automatically remove speech bubbles
        - Use **Right-to-left** reading order for traditional manga layout
        """)
        
        # Examples (optional)
        with gr.Accordion("🖼️ Example Images", open=False):
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
        
        gr.Markdown("# 🎨 Advanced Manhwa Panel Extractor")
        
        with gr.Tabs():
            # Single image processing tab
            with gr.TabItem("Single Image"):
                create_demo()
            
            # Batch processing tab
            with gr.TabItem("Batch Processing"):
                gr.Markdown("## 📚 Batch Process Multiple Images")
                
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
                gr.Markdown("## 🎯 Settings Presets")
                
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