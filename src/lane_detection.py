
import cv2
import numpy as np


def detect_lane_lines(roi_edges):
    """Detect left and right lane boundaries from ROI edge pixels."""

    height, width = roi_edges.shape[:2]

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    if lines is None:
        return None, None, 0

    lines = np.asarray(lines).reshape(-1, 4)

    left_points = []
    right_points = []

    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        slope = dy / dx
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        # Ignore nearly horizontal edges and extreme slopes.
        if abs(slope) < 0.5 or abs(slope) > 2.5:
            continue

        # Ignore edges near the top of the frame.
        if midpoint_y < height * 0.45:
            continue

        # Left lane: negative image-coordinate slope,
        # located on the left half of the road.
        if slope < 0 and midpoint_x < width * 0.52:
            left_points.extend([(x1, y1), (x2, y2)])

        # Right road edge: positive image-coordinate slope,
        # located toward the far-right side of the frame.
        elif slope > 0 and midpoint_x > width * 0.68:
            right_points.extend([(x1, y1), (x2, y2)])

    def fit_lane(points):
        if len(points) < 2:
            return None

        points = np.asarray(points, dtype=np.float64)

        # Fit x as a function of y.
        a, b = np.polyfit(points[:, 1], points[:, 0], 1)

        if not np.isfinite(a) or not np.isfinite(b):
            return None

        return np.array([a, b])

    left_lane = fit_lane(left_points)
    right_lane = fit_lane(right_points)

    return left_lane, right_lane, len(lines)


def draw_lane_lines(image, left_lane, right_lane):
    """Draw detected lane boundaries in green."""

    result = image.copy()
    height, width = image.shape[:2]

    # Keep the same drawing range as the original version.
    y_top = int(height * 0.55)
    y_bottom = height - 1

    for lane in (left_lane, right_lane):
        if lane is None:
            continue

        slope, intercept = lane

        x_top = int(slope * y_top + intercept)
        x_bottom = int(slope * y_bottom + intercept)

        # Clip the line correctly instead of independently
        # clamping its endpoints, which can distort its angle.
        visible, point1, point2 = cv2.clipLine(
            (0, 0, width, height),
            (x_top, y_top),
            (x_bottom, y_bottom)
        )

        if visible:
            cv2.line(
                result,
                point1,
                point2,
                (0, 255, 0),
                6,
                cv2.LINE_AA
            )

    return result
