import os
import time
import json
from PIL import Image
from google import genai
from google.genai import types

# Initialize Client (Assuming API Key is set in Env Vars)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# --- STEP 1: INTELLIGENT DIRECTOR (Gemini 1.5 Pro) ---
def get_smart_crop_box(image_path):
    img = Image.open(image_path)
    width, height = img.size
    target_width = int(height * (9 / 16))

    prompt = f"""
    Analyze this hotel room image. I need a 9:16 vertical crop.
    Identify the most visually premium subject (bed, view, or lounge).
    Return a bounding box JSON with keys: 'ymin', 'xmin', 'ymax', 'xmax'.
    The box width must be relative to {target_width} pixels.
    """

    # We use Flash for speed, or Pro for reasoning
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[prompt, img]
    )

    # In production: Use robust JSON parsing here
    # For demo, we assume the LLM returns the JSON block nicely
    text = response.text.replace("```json", "").replace("```", "")
    return json.loads(text)


# --- STEP 2: NANO BANANA PRO (Deterministic Matrix Crop) ---
def execute_precision_crop(image_path, bbox):
    img = Image.open(image_path)
    W, H = img.size

    # Extract & Validate Coordinates (Linear Algebra safety)
    # Ensuring we don't crash with out-of-bounds errors
    target_ratio = 9 / 16
    target_w = int(H * target_ratio)

    center_x = (bbox['xmin'] + bbox['xmax']) // 2

    # Calculate strict crop box centered on Gemini's suggestion
    left = max(0, center_x - (target_w // 2))
    right = left + target_w

    if right > W:
        right = W
        left = W - target_w

    # The Precision Cut
    crop = img.crop((left, 0, right, H))

    # Save temp file for Veo
    temp_path = "temp_crop_9_16.jpg"
    crop.save(temp_path)
    return temp_path


# --- STEP 3: VEO 3.1 (Generative Video) ---
def generate_video_stream(cropped_image_path):
    print("Uploading crop to Veo for latent conditioning...")

    # Upload the file to Gemini API storage so Veo can access it
    file_upload = client.files.upload(file=cropped_image_path)

    prompt = "Cinematic, slow pan, luxury hotel atmosphere, 4k, highly detailed, soft sunlight."

    print("Dreaming (Generating Video)...")

    # Call Veo 3.1
    # Note: 'image' parameter forces Image-to-Video mode
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",  # Redundant but good safety
            image=file_upload,  # <--- The Critical Conditioning Signal
        )
    )

    # Poll for completion (Video generation takes time)
    while not operation.done:
        print("Veo is rendering...")
        time.sleep(10)
        operation = client.operations.get(operation)

    # Retrieve Result
    generated_video = operation.result.generated_videos[0]
    return generated_video


# --- MAIN EXECUTION PIPELINE ---
def run_full_pipeline(source_image):
    # 1. Analyze
    print("Step 1: Gemini analyzing composition...")
    bbox = get_smart_crop_box(source_image)

    # 2. Crop
    print("Step 2: NB2 Pro cropping...")
    cropped_file = execute_precision_crop(source_image, bbox)

    # 3. Generate
    print("Step 3: Veo generating video...")
    video_result = generate_video_stream(cropped_file)

    # 4. Save
    print("Download complete.")
    # (Add download logic here based on 'video_result.video.uri')

run_full_pipeline("hotel_room.jpg")