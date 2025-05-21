def extend_bounding_box_area(image_width, image_height, x, y, w, h, extension_ratio=1.2):
    x -= int((extension_ratio - 1) * w / 2)
    y -= int((extension_ratio - 1) * h / 2)
    w = int(w * extension_ratio)
    h = int(h * extension_ratio)

    x = max(0, x)
    y = max(0, y)
    w = min(image_width - x, w)
    h = min(image_height - y, h)

    return x, y, w, h
