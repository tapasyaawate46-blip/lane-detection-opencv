# Lane Detection Using OpenCV for Driver Assistance

## Project Overview:
Lane Detection Using OpenCV for Driver Assistance is a computer vision project that identifies road lane boundaries from road images and video frames. It uses traditional image-processing techniques to highlight lane markings and demonstrate a basic lane-detection pipeline.
The project is implemented in Python using OpenCV and NumPy.

## Objectives
- Detect visible road lane boundaries from road images.
- Apply image preprocessing to improve edge detection.
- Identify the road region using a Region of Interest (ROI).
- Detect lane-line segments using the Probabilistic Hough Line Transform.
- Extend the lane-detection pipeline to video frames.

## Technologies Used
- Python
- OpenCV — image processing, edge detection and line detection
- NumPy — numerical operations and line fitting

## Methodology
The system follows these main steps:

1. Image input:Load a road image or read a frame from a video.
2. Grayscale conversion: Convert the colour image into grayscale.
3. Gaussian blur: Reduce noise before edge detection.
4. Canny edge detection: Identify strong edges that may correspond to road markings.
5. Region of Interest (ROI): Mask areas outside the selected road region.
6. Probabilistic Hough Line Transform: Detect candidate line segments from the ROI edge image.
7. Lane estimation: Group candidate segments into left and right lane boundaries and fit lines.
8. Output visualization: Draw the estimated lane boundaries in green on the original image or video frame.

For video processing, temporal smoothing is applied to reduce rapid changes in the estimated lane boundaries.

## Project Structure

lane-detection-opencv/
├── dataset/                 # Test road images
├── src/
│   ├── main.py              # Main execution script
│   ├── preprocessing.py     # Image preprocessing
│   ├── edge_detection.py    # Canny and ROI
│   ├── lane_detection.py    # Hough transform and lane fitting
│   ├── video_detection.py   # Video processing
│   └── output_utils.py      # Output saving utilities
├── video/
│   └── road_video.mp4       # Input road video
├── requirements.txt
└── README.md


## Installation
1. Install Python
2.Clone this repository:
   bash
   git clone https://github.com/tapasyaawate46-blip/lane-detection-opencv.git
   
3. Open the project folder:
   bash
   cd lane-detection-opencv
   
4. Create and activate a virtual environment if desired.
5. Install the required dependencies:
   bash
   pip install -r requirements.txt
  
## Running the Project
Run the main script from the project root:
bash
python -m src.main

The program processes the test images and, when the input video is available, processes the video as well. Generated results are saved in the configured output directory.

## Results
The implementation successfully detected lane boundaries in all 10 selected test images. The processed video demonstrates lane detection across consecutive frames, with temporal smoothing to reduce fluctuations.
Results depend on lighting, road markings, camera perspective and scene complexity.

## Limitations
- Curved roads and branching lanes can be difficult to represent with straight-line fitting.
- Shadows, worn road markings and other edges may affect detection.
- Lane estimates can flicker or become temporarily unavailable in video.
- The current implementation is a prototype and is not intended for direct vehicle control or safety-critical use.

## Future Improvements
- Improve lane tracking across consecutive video frames.
- Use curve-fitting methods for curved road boundaries.
- Evaluate performance on a larger and more diverse dataset.
- Explore deep-learning-based lane detection methods.

## Author
Developed as an academic project on computer vision and driver assistance.
