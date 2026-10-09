
import cv2
from pathlib import Path

from src.preprocessing import load_and_preprocess
from src.edge_detection import detect_edges, apply_region_of_interest
from src.lane_detection import detect_lane_lines, draw_lane_lines
from src.output_utils import save_image
from src.video_detection import process_video


def process_all_images(project_root):
    dataset_dir = project_root / "dataset"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support common image file extensions.
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    if not image_paths:
        print("No images found in the dataset folder.")
        return

    print(f"\nFound {len(image_paths)} images.")

    successful = 0

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\nProcessing image {index}/{len(image_paths)}: "
              f"{image_path.name}")

        try:
            image, gray, blurred = load_and_preprocess(
                str(image_path)
            )

            edges = detect_edges(blurred)
            roi_edges = apply_region_of_interest(edges)

            left_lane, right_lane, total_lines = detect_lane_lines(
                roi_edges
            )
            print("Left lane:", left_lane)
            print("Right lane:", right_lane)

            result = draw_lane_lines(
                image, left_lane, right_lane
            )

            # Give every image its own output filenames.
            save_image(
                edges,
                f"image_{index}_canny.jpg"
            )
            save_image(
                roi_edges,
                f"image_{index}_roi.jpg"
            )
            save_image(
                result,
                f"image_{index}_result.jpg"
            )

            print("Image dimensions:", image.shape)
            print("Detected line segments:", total_lines)
            print("Left boundary detected:", left_lane is not None)
            print("Right boundary detected:", right_lane is not None)

            successful += 1

        except (FileNotFoundError, cv2.error, ValueError) as error:
            print(f"Could not process {image_path.name}: {error}")

    print("\n--- Image Processing Summary ---")
    print(f"Images found: {len(image_paths)}")
    print(f"Images processed successfully: {successful}")
    print(f"Results saved in: {output_dir}")


def main():
    # Project root is the parent of the src folder.
    project_root = Path(__file__).resolve().parent.parent

    # Step 1: Process all dataset images.
    process_all_images(project_root)

    # Step 2: Process the road video separately.
    video_path = project_root / "video" / "road_video.mp4"
    output_path = (
        project_root / "outputs" / "processed_road_video.mp4"
    )

    if video_path.is_file():
        print("\nStarting video processing...")
        process_video(str(video_path), str(output_path))
    else:
        print("\nVideo not found; skipping video processing.")


if __name__ == "__main__":
    main()
