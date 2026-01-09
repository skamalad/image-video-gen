
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

