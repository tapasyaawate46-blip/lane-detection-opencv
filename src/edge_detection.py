
import cv2
import numpy as np


def detect_edges(blurred_image):
    """
    Detect edges in a blurred grayscale image using Canny.
    """
    edges = cv2.Canny(
        blurred_image,
        threshold1=50,
        threshold2=150
    )

    return edges


def apply_region_of_interest(edges):
    """
    Keep only the road region using a polygon that adapts
    to the input image dimensions.
    """
    height, width = edges.shape[:2]

    # Create a black mask matching the input dimensions.
    mask = np.zeros_like(edges)

    # Define a trapezoidal region around the road.
    polygon = np.array(
        [[
            (0, height - 1),
            (width - 1, height - 1),
            (int(width * 0.78), int(height * 0.49)),
            (int(width * 0.27), int(height * 0.49))
        ]],
        dtype=np.int32
    )

    # Fill the road region with white.
    cv2.fillPoly(mask, polygon, 255)

    # Keep edges only inside the selected road region.
    roi_edges = cv2.bitwise_and(edges, mask)

    return roi_edges
