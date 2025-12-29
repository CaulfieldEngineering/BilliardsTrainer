#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

// Structure to hold rail detection parameters
struct RailParams {
    // Rail detection parameters
    int railBandWidth = 30;           // Width of band around felt edge to search for rails (pixels)
    int railMaxWidth = 20;            // Maximum width of rail from felt edge (pixels) - constrains rail thickness
    int railMinArea = 100;            // Minimum area for a valid rail region (pixels^2)
    int railEdgeThreshold = 50;       // Canny edge threshold for rail detection
    int railEdgeThreshold2 = 150;     // Canny edge threshold (high) for rail detection

    // Color-based filtering (for rail wood color detection)
    bool useColorFilter = true;       // Whether to use color filtering for rails (enabled by default when color picked)
    cv::Scalar railColorMin = cv::Scalar(0, 0, 0);    // BGR minimum (populated from HSV range)
    cv::Scalar railColorMax = cv::Scalar(255, 255, 255);  // BGR maximum (populated from HSV range)

    // Color picker state (similar to FeltParams)
    bool hasPickedColor = false;      // Whether user has picked a color via click-to-sample
    cv::Vec3b pickedBGR = cv::Vec3b(139, 69, 19);  // Picked BGR (default brown)
    cv::Vec3b pickedHSV = cv::Vec3b(10, 191, 139);  // Picked HSV (converted from default brown)
    int colorSensitivity = 35;        // 0..100, controls HSV tolerance
    // HSV range computed from picked color + sensitivity
    int colorHMin = 0;
    int colorHMax = 179;
    int colorSMin = 0;
    int colorSMax = 255;
    int colorVMin = 0;
    int colorVMax = 255;

    // Overlay styling
    cv::Scalar color = cv::Scalar(139, 69, 19);  // BGR brown (typical rail color)
    bool isFilled = true;
    int fillAlpha = 100;              // 0..255 (only used if isFilled)
    int outlineThicknessPx = 2;       // used if !isFilled (or for outline)
};

// Result structure for rail detection
struct RailDetectionResult {
    cv::Mat railMask;                 // CV_8U, 0/255 - combined mask of all rail surfaces
    std::vector<cv::Mat> railMasks;   // Individual masks for each rail (top, right, bottom, left)
    std::vector<std::vector<cv::Point>> railOutlines;  // Outlines of each rail
    std::vector<cv::Rect> railBoundingBoxes;  // Bounding boxes for each rail
    bool ok;                          // True if detection passed validation checks
    int numRailsDetected;             // Number of rails successfully detected (0-4)
};

// Main rail detection function - works on rectified view
// Inputs:
//   rectifiedBgr: The rectified BGR image
//   rectifiedFeltMask: The rectified felt mask (binary, 0/255)
//   params: Detection parameters
RailDetectionResult detectRails(
    const cv::Mat& rectifiedBgr,
    const cv::Mat& rectifiedFeltMask,
    const RailParams& params
);

// Project rail detection results from rectified view back to perspective view
// Inputs:
//   rectifiedResult: Rail detection result in rectified space
//   Hinv: Inverse homography matrix (3x3 CV_64F) to transform from rectified to perspective
//   perspectiveSize: Size of the perspective view image
// Returns:
//   RailDetectionResult with all points/masks projected to perspective view
RailDetectionResult projectRailsToPerspective(
    const RailDetectionResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
);

// Draw rail overlay on an image (in-place modification)
// Applies the rail mask overlay with styling from params
void drawRailOverlay(cv::Mat& img, const RailDetectionResult& result, const RailParams& params);

// Draw rail outlines on an image (in-place modification)
// Draws the outlines of each detected rail with styling from params
void drawRailOutlines(cv::Mat& img, const RailDetectionResult& result, const RailParams& params);

