import os
import time
import json
import utils
from PIL import Image
from google import genai
from google.genai import types
import execute_precision_crop
import generate_video_stream

# Initialize Client (Assuming API Key is set in Env Vars)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# Step 1
def get_smart_crop_box(image_path):
    img = Image.open(image_path)
    width, height = img.size
    target_width = int(height * (9 / 16))

    prompt = f"""
    Analyze this image. I need a 9:16 vertical crop.
    Identify the most visually premium subject (bed, view, or lounge).
    Return a bounding box JSON with keys: 'ymin', 'xmin', 'ymax', 'xmax'.
    The box width must be relative to {target_width} pixels.
    """

    # We use Flash for speed, or Pro for reasoning
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[prompt, img]
    )

    text = response.text.replace("```json", "").replace("```", "")
    return json.loads(text)

def parse_gemini_bbox(gemini_response):
    """
    Parse gemini response into a standard format
    Convert to [xmin, xmax, ymin, ymax]
    Args:
        gemini_response: raw list from gemini
    Returns:
        dict with normalized coordinates in named format
    """

    detection = gemini_response[0]
    box = detection['box_2d']

    return {
        'ymin': box[0],
        'xmin': box[1],
        'ymax': box[2],
        'xmax': box[3],
        'label': detection.get('label', 'unknown')
    }



# --- MAIN EXECUTION PIPELINE ---
def run_full_pipeline(source_image):
    # 1. Analyze
    print("Step 1: Gemini analyzing composition...")
    bbox = get_smart_crop_box(source_image)

    # 2. Crop
    print("Step 2: NB2 Pro cropping...")
    cropped_file = (execute_precision_crop.execute_precision_crop(source_image, bbox))

    # 3. Generate
    print("Step 3: Veo generating video...")
    video_result = generate_video_stream.generate_video_stream(cropped_file)

    # 4. Save
    print("Download complete.")
    # (Add download logic here based on 'video_result.video.uri')

run_full_pipeline("hotel_room.jpg")