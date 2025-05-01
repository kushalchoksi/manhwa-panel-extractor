# Manhwa Panel Extractor

Scrolling through manhwa pages on your phone is a pain, there's just too much white space. Plus I sometimes like to make memes from just the image. 

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
    python manhwa-extractor.py jinwoo.jsp
    ```
4. **Extracts panels** in a folder of the image or directory name with `_output` appended 

There's a few other options to extract it in different ways if you want to play around with that. 
