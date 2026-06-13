"""
segmenter.py
------------
Line segmentation for handwritten text pages.

Splits a page image into individual line images so TrOCR (and EasyOCR)
can process them line by line — which is dramatically more accurate than
passing a full page.

Strategy:
  1. Binarize (inverse — text is white on black)
  2. Project horizontal histogram — count white pixels per row
  3. Find gaps (rows with few white pixels) = line boundaries
  4. Crop + return each line as a PIL Image with padding

Usage:
  from segmenter import segment_lines
  lines = segment_lines(pil_image)  # list of PIL Images
"""

import numpy as np
from PIL import Image
import cv2
import logging


def segment_lines(img: Image.Image, 
                  min_line_height: int = 15,
                  gap_threshold: float = 0.02,
                  padding: int = 4) -> list:
    """
    Segment a page image into line images.

    Args:
        img: PIL Image of the page (preprocessed or raw)
        min_line_height: ignore segments shorter than this (px) — filters noise
        gap_threshold: fraction of max row density below which a row is a gap
        padding: pixels to add above/below each line crop

    Returns:
        List of PIL Images, one per text line (top to bottom)
    """
    gray = np.array(img.convert("L"))

    # Binarize — invert so text = white (255), background = black (0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum white pixels per row
    projection = np.sum(binary, axis=1) / 255  # number of white pixels per row

    if projection.max() == 0:
        logging.warning("[Segmenter] blank page — no lines detected")
        return [img]

    threshold = projection.max() * gap_threshold

    # Find line regions: rows above threshold
    in_line = projection > threshold
    lines = []
    start = None

    for i, active in enumerate(in_line):
        if active and start is None:
            start = i
        elif not active and start is not None:
            end = i
            if (end - start) >= min_line_height:
                lines.append((start, end))
            start = None
    if start is not None:
        lines.append((start, len(in_line)))

    if not lines:
        logging.warning("[Segmenter] no lines found — returning full image")
        return [img]

    logging.debug(f"[Segmenter] found {len(lines)} lines")

    # Crop with padding
    h, w = gray.shape
    crops = []
    for y1, y2 in lines:
        top    = max(0, y1 - padding)
        bottom = min(h, y2 + padding)
        crop = img.crop((0, top, w, bottom))
        crops.append(crop)

    return crops
