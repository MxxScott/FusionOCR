import pytesseract
import easyocr
import Levenshtein
import os
import logging
import functools
import time
from PIL import Image
from transformers import pipeline
from calamari_ocr.ocr import Predictor
from datetime import datetime


# Get the folder where the Python file resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create log directory next to the Python file
log_dir = os.path.join(script_dir, "ocr_logs")
os.makedirs(log_dir, exist_ok=True)

nlp = pipeline("text2text-generation", model="google/flan-t5-base",
               device=0)  # Use GPU if available
# nlp = pipeline("text2text-generation", model="google/flan-t5-small",
#              device=0)  # Use GPU if available
easyocr_reader = easyocr.Reader(
    # Initialize once})
    ['en'], gpu=True, model_storage_directory="models", download_enabled=True)


def log_function(func):
    """Decorator to log entry and exit of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"[{func.__name__}] started")
        try:
            result = func(*args, **kwargs)
            logging.debug(f"[{func.__name__}] finished")
            return result
        except Exception as e:
            logging.error(f"[{func.__name__}] error: {e}", exc_info=True)
            raise
    return wrapper


# Setup logging (every run creates a new log file)
@log_function
def setup_logging():
    log_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "ocr_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("ocr_run_%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),  # full debug
            logging.StreamHandler()  # only higher levels
        ],
        force=True
    )
    logging.getLogger().handlers[1].setLevel(logging.INFO)  # console = INFO+
    return log_path


log_path = setup_logging()
print(f"Logs will be saved to: {log_path}")


@log_function
def load_image(image_path):
    # """Load an image from the specified path."""
    return Image.open(image_path)


@log_function
def pytesseract_ocr(image_path):
    # Load image for Tesseract
    image = load_image(image_path)

    # """Extract text from an image using pytesseract."""
    text = pytesseract.image_to_string(image)
    logging.debug(f"[Pytesseract] OCR result: {text}")
    return text


@log_function
def easyocr_ocr(image_path):
    # """Extract text from an image using EasyOCR."""
    result = easyocr_reader.readtext(image_path, detail=0)
    text = ' '.join(result)
    logging.debug(f"[EasyOCR] OCR result: {text}")
    return text


@log_function
def calamari_ocr(image_path):
    model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "uw3-modern-english")
    logging.info("[Calamari] Starting OCR")
    
    predictor = Predictor(model_path)
    predictions = predictor.predict([image_path])
    text = ' '.join([p.sentence for p in predictions])
    logging.debug(f"[Calamari] OCR result: {text}")
    return text


@log_function
def consensus_word(words):
    # """Determine the consensus word from a list of words using Levenshtein distance."""
    if not words:
        return ""

    def total_distance(word):
        return sum(Levenshtein.distance(word, other) for other in words if word != other)

    return min(words, key=total_distance)


@log_function
def consensus_line(lines):
    # """Determine the consensus line from a list of lines."""
    if not lines:
        return ""

    split_lines = [line.split() for line in lines]
    max_length = max(len(line) for line in split_lines)

    consensus_words = []
    for i in range(max_length):
        words_at_position = [line[i] for line in split_lines if i < len(line)]
        consensus_words.append(consensus_word(words_at_position))

    return ' '.join(consensus_words)


@log_function
def text_cleanup(text):
    # """Clean up the extracted text using a language model."""
    cleaned_text = nlp(text, max_length=512, do_sample=False)[
        0]['generated_text']
    return cleaned_text


@log_function
def run_ocr(image_path):
    start_time = time.time()
    logging.info(f"Running OCR pipeline on {image_path}")

    # Run OCR engines
    tesseract_text = pytesseract_ocr(image_path)
    easyocr_text = easyocr_ocr(image_path)
    calamari_text = calamari_ocr(image_path)

    # Combine with consensus
    combined_text = consensus_line(
        [tesseract_text, easyocr_text, calamari_text])
    logging.info(f"[Consensus] Result: {combined_text}")

    print("\n--- Consensus Result ---")
    print(combined_text)

    # Clean up with language model
    cleaned_text = text_cleanup(combined_text)
    logging.info(f"[Cleanup] Result: {cleaned_text}")

    print("\n--- Cleaned Result ---")
    print(cleaned_text)

    # --- Summary ---
    elapsed = time.time() - start_time
    logging.info("=" * 50)
    logging.info("OCR PIPELINE SUMMARY")
    logging.info(f"Image processed : {image_path}")
    logging.info(f"Engines used    : pytesseract, easyocr, calamari")
    logging.info(f"Consensus text  : {combined_text}")
    logging.info(f"Cleaned text    : {cleaned_text}")
    logging.info(f"Elapsed time    : {elapsed:.2f} sec")
    logging.info(f"Log file saved  : {log_path}")
    logging.info("=" * 50)

    # Print summary to console too
    print("\n" + "=" * 50)
    print("OCR PIPELINE SUMMARY")
    print(f"Image processed : {image_path}")
    print(f"Engines used    : pytesseract, easyocr, calamari")
    print(f"Consensus text  : {combined_text}")
    print(f"Cleaned text    : {cleaned_text}")
    print(f"Elapsed time    : {elapsed:.2f} sec")
    print(f"Log file saved  : {log_path}")
    print("=" * 50 + "\n")

    print(f"\nLog saved to: {log_path}")
    return cleaned_text


# Example usage
if __name__ == "__main__":
    test_image_path = os.path.join(script_dir, "test_image.png")
    if os.path.exists(test_image_path):
        run_ocr(test_image_path)
    else:
        print(
            f"Test image not found at {test_image_path}. Please provide a valid image path.")
