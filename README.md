# To Buld
cd C:\Users\jpcfo\Documents\_GitHub\_CaulfieldEngineering\BilliardsTrainer\build; cmake --build . --config Release


# To-Do
1) Felt Dection
2) Rail Detection
   - Rudimentary success
   - Uses Felt Detection to subtract felt
3) Diamond Detection
   - Uses Rail detection bounds to know where to look

* TBD
- Grid Extrapolation of Diamonds
- Playing Field Definition based on Felt
- Detection / Overlay of pockets

# Billiards Trainer - Table Detection Proof of Concept

A computer vision application that detects a billiards table in real-time video and overlays visualizations for the felt, rails, pockets, and diamonds.

## Features

- **Felt Detection**: Automatically detects the playing surface using color-based segmentation (supports both green and blue felt)
- **Rail Detection**: Identifies and visualizes the four cushion rails around the table
- **Pocket Detection**: Detects the six pockets (4 corners + 2 side pockets) using circular Hough transform
- **Diamond Detection**: Identifies and marks the diamond markers on each rail
- **Real-time Overlay**: Draws semi-transparent overlays on detected elements

## Requirements

- OpenCV 4.x
- CMake 3.16 or higher
- C++17 compatible compiler
- Webcam or video file input

## Building

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage

### Camera Input (Default Camera)
```bash
.\build\Release\table_detector.exe
```

### Camera Input (Specific Camera Index)
```bash
.\build\Release\table_detector.exe 0    # Use camera 0
.\build\Release\table_detector.exe 1    # Use camera 1
.\build\Release\table_detector.exe 2    # Use camera 2
```

### Video File Input
```bash
.\build\Release\table_detector.exe path/to/video.mp4
```

### Image Input
```bash
.\build\Release\table_detector.exe path/to/image.jpg
```

## Controls

- **'q' or ESC**: Quit the application
- **'s'**: Save current frame with overlay to disk
- **'r'**: Reset detection (for future calibration features)
- **'c'**: Cycle through available cameras
- **'0'-'9'**: Switch to camera 0-9 directly

## Detection Algorithm

1. **Felt Detection**: Uses HSV color space to segment green or blue felt from the background
2. **Corner Detection**: Finds the four corners of the felt using contour approximation
3. **Pocket Detection**: Uses Hough circle transform to detect circular pockets
4. **Rail Detection**: Calculates rail positions based on felt corners and extends outward
5. **Diamond Detection**: Places diamond markers at standard positions along each rail

## Overlay Visualization

- **Green Overlay**: Semi-transparent overlay on the felt surface
- **Orange Lines**: Rail boundaries
- **Red Circles**: Detected pockets
- **Yellow Diamonds**: Rail markers (diamonds)
- **Magenta Circles**: Corner markers (for debugging)

## Notes

- The camera should be stable on a tripod for best results
- Good lighting conditions improve detection accuracy
- The algorithm works best when the table fills a significant portion of the frame
- Detection parameters may need adjustment based on camera angle and table size

## Future Enhancements

- Calibration mode for fine-tuning detection parameters
- Ball tracking and trajectory prediction
- Shot analysis and recommendations
- Multi-table support
- Improved diamond detection using template matching
- Perspective correction for accurate measurements
