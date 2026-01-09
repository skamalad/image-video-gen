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

    # text = response.text.replace("```json", "").replace("```", "")
    # return json.loads(text)
    text = response.text.replace("```json", "").replace("```", "")
    parsed = json.loads(text)
    print(f"DEBUG = Gemini returned: {parsed}")
    print(f"DEBUG - Type: {type(parsed)}")
    return parsed


# Step 3
def generate_video_stream(cropped_image_path):
    print("Uploading crop to Veo for latent conditioning...")

    # Upload the file to Gemini API storage so Veo can access it
    # file_upload = client.files.upload(file=cropped_image_path)
    with open(cropped_image_path, "rb") as f:
        image_bytes = f.read()

    if cropped_image_path.endswith(".jpg") or cropped_image_path.endswith(".jpeg"):
        mime_type="image/jpeg"
    elif cropped_image_path.endswith(".png"):
        mime_type="image/png"
    else:
        mime_type="image/jpeg"

    image = types.Image(image_bytes=image_bytes, mime_type=mime_type)

    prompt = "Cinematic, slow pan, luxury hotel atmosphere, 4k, highly detailed, soft sunlight."

    print("Dreaming (Generating Video)...")

    # Call Veo 3.1
    # Note: 'image' parameter forces Image-to-Video mode
    # print(types.GenerateVideosConfig.model_fields.keys())

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        image=image,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16", # Redundant but good safety
        )
    )

    # Poll for completion (Video generation takes time)
    while not operation.done:
        print("Veo is rendering...")
        time.sleep(10)
        operation = client.operations.get(operation)

    # Retrieve Result
    generated_video = operation.response.generated_videos[0]
    return generated_video


# --- MAIN EXECUTION PIPELINE ---
def run_full_pipeline(source_image):
    # 1. Analyze
    print("Step 1: Gemini analyzing composition...")
    bbox_list = get_smart_crop_box(source_image)
    bbox_dict = utils.parse_gemini_bbox(bbox_list)

    # 2. Crop
    print("Step 2: NB2 Pro cropping...")
    cropped_file = execute_precision_crop.execute_precision_crop(source_image, bbox_dict)

    # 3. Generate
    print("Step 3: Veo generating video...")
    video_result = generate_video_stream(cropped_file)

    # 4. Save
    print("Downloading video")
    client.files.download(file=video_result.video)
    output_filename = "generated_video_9_16.mp4"
    video_result.video.save(output_filename)
    print(f"Download complete. Video saved to {output_filename}")
    # (Add download logic here based on 'video_result.video.uri')

run_full_pipeline("hotel_room.jpg")