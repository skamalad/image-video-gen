
# Step 3
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
            # aspect_ratio="9:16",  # Redundant but good safety
            image=file_upload,
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
