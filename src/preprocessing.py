import cv2


def load_and_preprocess(image_path):
    """
    Load an image and perform basic preprocessing.

    Steps:
    1. Load image
    2. Convert BGR image to grayscale
    3. Apply Gaussian blur
    """

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return image, gray, blur