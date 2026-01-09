
def denormalize_bbox(bbox, image_width, image_height, scale=1000):
    """
    Convert normalized bounding box coordinates to pixel coordinates

    Args:
        bbox: dict with 'xmin', 'xmax', 'ymin', 'ymax' in normalized form (0-1000)
        image_width: actual image width in pixels
        image_height: actual image height in pixels
        scale: the normalization scale

    Returns:
        dict with pixel coordinates
    """
    return {
        'xmin': int((bbox['xmin'] / scale) * image_width),
        'xmax': int((bbox['xmax'] / scale) * image_width),
        'ymin': int((bbox['ymin'] / scale) * image_height),
        'ymax': int((bbox['ymax'] / scale) * image_height),
    }

def parse_gemini_bbox(gemini_response):
    """
    Parse gemini response into a standard format
    Handle multiple response formats since we are using free form prompting
    Convert to [xmin, xmax, ymin, ymax]
    Args:
        gemini_response: raw list from gemini
    Returns:
        dict with normalized coordinates in named format
    """

    if isinstance(gemini_response, dict):
        if all(key in gemini_response for key in['ymin', 'xmin', 'ymax', 'xmax']):
            return gemini_response

    if isinstance(gemini_response, list) and len(gemini_response) > 0:
        detection = gemini_response[0]
        box = detection['box_2d']

        return {
            'ymin': box[0],
            'xmin': box[1],
            'ymax': box[2],
            'xmax': box[3],
            'label': detection.get('label', 'unknown')
        }

    raise ValueError(f"Unexpected response format: {type(gemini_response)} - {gemini_response}")

