from PIL import Image
import utils


# Step 2
def execute_precision_crop(image_path, bbox):
    img = Image.open(image_path)
    W, H = img.size

    source_ratio = W / H
    target_ratio = 9 / 16

    pixel_bbox = utils.denormalize_bbox(bbox, W, H, scale=1000)

    if source_ratio > target_ratio:
        target_w = int(H * target_ratio)
        target_h = H
        center_x = (pixel_bbox['xmin'] + pixel_bbox['xmax']) // 2

        # Calculate strict crop box centered on Gemini's suggestion
        left = max(0, center_x - (target_w // 2))
        right = left + target_w

        if right > W:
            right = W
            left = W - target_w

        top = 0
        bottom = H

    elif source_ratio < target_ratio:
        target_w = W
        target_h = int(W / target_ratio)

        if target_h > H:
            # cannot crop at full width, image is too short
            target_h = H
            target_w = int(H * target_ratio)

            center_x = (pixel_bbox['xmin'] + pixel_bbox['xmax']) // 2
            left = max(0, center_x - (target_w // 2))
            right = left + target_w
            if right > W:
                right = W
                left = W - target_w
            top = 0
            bottom = H
        else:
            center_y = (pixel_bbox['ymin'] + pixel_bbox['ymax']) // 2
            top = max(0, center_y - (target_h // 2))
            bottom = top + target_h

            if bottom > H:
                bottom = H
                top = H - target_h

            left = 0
            right = W
    else:
        left, top, right, bottom = 0, 0, W, H

    # precision cut
    crop = img.crop((left, top, right, bottom))

    # Save temp file for Veo
    temp_path = "temp_crop_9_16.jpg"
    crop.save(temp_path)
    return temp_path


