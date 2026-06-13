"""
preprocessor.py
---------------
Image preprocessing pipeline for handwritten text OCR.
Tuned for children's handwriting on lined/grid paper.

Steps:
  1. Load image
  2. Convert to grayscale
  3. Remove ruled lines (horizontal + vertical)
  4. Deskew
  5. Denoise
  6. Adaptive binarize (handles uneven phone-photo lighting)
  7. Mild morphological cleanup (fills small ink gaps)

Usage:
  from preprocessor import preprocess
  img = preprocess("scan.jpg")
"""

import numpy as np
from PIL import Image
import cv2
import logging


def _remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """
    Detect and remove horizontal ruled lines and vertical margin lines
    that appear on exercise book / exam script paper.
    Uses morphological operations to isolate and subtract lines.
    """
    # Binarize for line detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # --- Horizontal lines ---
    # Kernel wider than any single character stroke
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 15, 40), 1))
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Dilate slightly to cover line thickness variation
    h_dilate = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    # --- Vertical margin line (left red margin) ---
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 15, 40)))
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_dilate = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

    # Combine line masks
    line_mask = cv2.bitwise_or(h_dilate, v_dilate)

    # Remove lines from binary (invert back, paint out lines)
    cleaned = cv2.bitwise_not(binary)
    cleaned[line_mask == 255] = 255  # set line pixels to white (background)

    logging.debug("[Preprocessor] ruled lines removed")
    return cleaned


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct tilt up to ±15° using Hough line detection."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=80, minLineLength=80, maxLineGap=10)
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    angles = [a for a in angles if abs(a) < 15]
    if not angles:
        return gray

    angle = np.median(angles)
    logging.debug(f"[Preprocessor] deskew {angle:.2f}°")

    ch, cw = gray.shape
    M = cv2.getRotationMatrix2D((cw // 2, ch // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (cw, ch),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding — critical for phone photos with shadows.
    Larger block size handles the lighting gradients in photographed scripts.
    """
    # CLAHE (better than equalizeHist for uneven lighting)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=4,
    )
    return binary


def _morph_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Light closing to fill small gaps in ink strokes
    (common in children's writing with varying pen pressure).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def preprocess(source, binarize: bool = True,
               remove_lines: bool = True) -> Image.Image:
    """
    Full preprocessing pipeline.

    Args:
        source:       file path (str) or PIL Image
        binarize:     output black-and-white (recommended for OCR)
        remove_lines: strip ruled/lined paper lines before OCR

    Returns:
        Preprocessed PIL Image (mode 'L')
    """
    if isinstance(source, str):
        img = Image.open(source)
    else:
        img = source.copy()

    gray = np.array(img.convert("L"))

    if remove_lines:
        gray = _remove_ruled_lines(gray)
    else:
        gray = gray

    gray = _deskew(gray)
    gray = _denoise(gray)

    if binarize:
        gray = _binarize(gray)
        gray = _morph_cleanup(gray)

    return Image.fromarray(gray)
