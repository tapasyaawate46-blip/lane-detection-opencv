
import os
import cv2


def save_image(image, filename):
    """Save a processed image in the outputs folder."""

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    success = cv2.imwrite(output_path, image)

    if success:
        print(f"Image saved successfully: {output_path}")
    else:
        print(f"Error saving image: {output_path}")
