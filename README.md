# Manhwa Panel Extractor

Scrolling through manhwa pages on your phone is a pain, there's just too much white space. Plus I sometimes like to make memes from just the image. 

![image](https://github.com/user-attachments/assets/7d1e0004-89f8-45d2-8604-7da116c3d630)

# Demo

Try it out yourself here https://huggingface.co/spaces/kushalchoksi/manhwa-panel-extractor 

## What It Does
- **Extracts Panels**: Slices manhwa pages into individual panels.
- **Refines Them**: Removes speech bubbles using HSV magic.


## Setup
1. **Get Python**: 3.8+
2. **Install the requirements**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **Run** the main extractor on an image or directory
    ```bash
    python manhwa-extractor.py jinwoo.jpg
    ```
4. **Extracts panels** in a folder of the image or directory name with `_output` appended 

There's a few other options to extract it in different ways if you want to play around with that.

## Web Interface

You can also use the Gradio web interface if you're not good with CLI's:

```bash
python gradio_app.py
```

This launches a local web app at `http://localhost:7860` where you can:
- Upload images through your browser
- Adjust extraction settings with sliders
- Preview extracted panels in a gallery
- Download all panels as a ZIP file

## Potential Improvements

- Have a way of doing automatic inpainting on leftover/partial speech bubbles
