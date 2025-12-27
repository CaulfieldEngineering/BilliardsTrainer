#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

// We accept these parameter structs so the sidebar tuning actually impacts diamond detection.
#include "../FeltDetection/felt_detection.h"
#include "../RailDetection/rail_detection.h"

// Structure to hold detection parameters
struct DiamondDetectionParams {
    // Detection algorithm parameters
    int min_threshold = 30;           // Threshold for enhanced image (0-255). Lower values = more sensitive, higher values = less sensitive.
    int minArea = 20;                 // Minimum contour area in pixels
    int maxArea = 300;                // Maximum contour area in pixels
    float min_circularity = 0.4f;     // Minimum circularity (0.0-1.0), higher = more circular
    int morph_kernel_size = 15;       // Morphological kernel size for top-hat/black-hat (should be larger than diamond size)
    bool skip_morph_enhancement = false;  // If true, skip morphological enhancement and threshold grayscale directly
    int adaptive_thresh_blocksize = 11;   // Block size for adaptive thresholding (must be odd: 3, 5, 7, 9, 11, etc.)
    int adaptive_thresh_C = 2;            // Constant subtracted from mean for adaptive thresholding
    
    // Color filtering (HSV range) - if enabled, only detect diamonds matching this color
    // Color filtering is now always ON (UI toggle removed). Keep the flag for backward compatibility.
    bool use_color_filter = true;         // Always true; UI no longer exposes disabling this.
    int colorHMin = 0, colorHMax = 180;   // Hue range (0-180 in OpenCV)
    int colorSMin = 0, colorSMax = 255;   // Saturation range (0-255)
    // Default to a no-op filter (full V range) until a color is picked.
    int colorVMin = 0, colorVMax = 255;   // Value range (0-255)

    // Color picker state (UI convenience):
    // - `pickedHSV` is stored so changing the sensitivity slider can recompute the min/max ranges
    //   without requiring the user to re-pick the color.
    // - Hue wrapping is represented by allowing colorHMin > colorHMax (range spans 180->0).
    bool hasPickedColor = false;
    cv::Vec3b pickedHSV = cv::Vec3b(0, 0, 0);

    // Sensitivity / tolerance (0..100):
    // - 0  => very strict (only very close matches)
    // - 100 => very loose (accepts more variation from the picked color)
    int colorSensitivity = 50;
    cv::Vec3b pickedBGR = cv::Vec3b(0, 0, 0);  // Store the picked BGR color for UI display
    
    // Legacy parameters (kept for UI compatibility, threshold1 maps to min_threshold)
    int threshold1 = 100;             // Maps to min_threshold for backward compatibility
    int threshold2 = 255;             // Unused but kept for UI compatibility
    
    // Overlay styling
    cv::Scalar color = cv::Scalar(0, 0, 255); // BGR (red by default for diamonds)
    // Alpha (transparency) for rendering: 0..255 (0 = invisible, 255 = fully opaque)
    int alpha = 255;
    bool isFilled = true;
    int radiusPx = 8;                 // Radius of drawn diamond markers
    int outlineThicknessPx = 2;       // Outline thickness for markers
};

// Debug images emitted by the diamond detection pipeline (label, image)
extern std::vector<std::pair<std::string, cv::Mat>> g_lastDiamondDebugImages;

// Detect diamond markers on the pool table rails with context-aware constraints
void detectDiamonds(
    const cv::Mat& src,
    cv::Mat& dst,
    bool showDiamonds,
    const DiamondDetectionParams& diamondParams,
    const FeltParams& feltParams,
    const RailParams& railParams,
    cv::Mat* outProcessingImage = nullptr);
