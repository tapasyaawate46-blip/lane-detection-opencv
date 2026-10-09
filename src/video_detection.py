
import cv2
import os
import time

from src.edge_detection import detect_edges, apply_region_of_interest
from src.lane_detection import detect_lane_lines, draw_lane_lines


def preprocess_frame(frame):
    """Convert a video frame to grayscale and reduce noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def process_video(video_path, output_path):
    """Detect lane boundaries and smooth them across video frames."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Resize frames for faster processing.
    width, height = 640, 360

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_count = 0
    start_time = time.perf_counter()

    previous_left = None
    previous_right = None
    alpha = 0.25

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Resize before applying edge and lane detection.
            frame = cv2.resize(frame, (width, height))

            # 1. Preprocessing
            blurred = preprocess_frame(frame)

            # 2. Edge detection and ROI
            edges = detect_edges(blurred)
            roi_edges = apply_region_of_interest(edges)

            # 3. Detect lane lines
            left_lane, right_lane, _ = detect_lane_lines(roi_edges)

            # 4. Smooth lane estimates between frames
            if left_lane is not None:
                if previous_left is not None:
                    left_lane = (
                        alpha * left_lane
                        + (1 - alpha) * previous_left
                    )
                previous_left = left_lane

            if right_lane is not None:
                if previous_right is not None:
                    right_lane = (
                        alpha * right_lane
                        + (1 - alpha) * previous_right
                    )
                previous_right = right_lane

            # Use the previous estimate if a lane is briefly missed.
            display_left = (
                left_lane if left_lane is not None else previous_left
            )
            display_right = (
                right_lane if right_lane is not None else previous_right
            )

            # 5. Draw lane boundaries
            result = draw_lane_lines(
                frame, display_left, display_right
            )

            frame_count += 1

            cv2.putText(
                result,
                f"Frame: {frame_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            writer.write(result)

            if frame_count % 25 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"Processed {frame_count} frames "
                    f"in {elapsed:.1f} seconds"
                )

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start_time

    print("\n--- Video Processing Summary ---")
    print(f"Processed frames: {frame_count}")
    print(f"Processing time: {elapsed:.2f} seconds")

    if elapsed > 0:
        print(f"Average processing speed: {frame_count / elapsed:.2f} FPS")

    print(f"Output saved to: {output_path}")
