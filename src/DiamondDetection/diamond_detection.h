#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

// Structure to hold diamond detection parameters
struct DiamondParams {
    // Diamond detection parameters
    int minArea = 50;                 // Minimum area for a valid diamond (pixels^2)
    int maxArea = 500;                // Maximum area for a valid diamond (pixels^2)
    int edgeThreshold = 50;           // Canny edge threshold for diamond detection
    int edgeThreshold2 = 150;         // Canny edge threshold (high) for diamond detection

    // Expected diamond counts per rail
    int shortRailDiamonds = 3;        // Number of diamonds on short rails (top/bottom)
    int longRailDiamonds = 6;         // Number of diamonds on long rails (left/right)

    // Spatial constraints
    float minDiamondSpacingPx = 30.0f;  // Minimum spacing between diamonds along rail axis (rectified space)

    // Color-based filtering (for diamond color detection)
    bool useColorFilter = true;       // Whether to use color filtering for diamonds
    cv::Scalar diamondColorMin = cv::Scalar(0, 0, 0);    // BGR minimum
    cv::Scalar diamondColorMax = cv::Scalar(255, 255, 255);  // BGR maximum

    // Color picker state (similar to FeltParams/RailParams)
    bool hasPickedColor = false;      // Whether user has picked a color via click-to-sample
    cv::Vec3b pickedBGR = cv::Vec3b(255, 255, 255);  // Picked BGR (default white)
    cv::Vec3b pickedHSV = cv::Vec3b(0, 0, 255);      // Picked HSV (converted from default white)
    int colorSensitivity = 35;        // 0..100, controls HSV tolerance
    // HSV range computed from picked color + sensitivity
    int colorHMin = 0;
    int colorHMax = 179;
    int colorSMin = 0;
    int colorSMax = 255;
    int colorVMin = 0;
    int colorVMax = 255;

    // Overlay styling
    cv::Scalar color = cv::Scalar(255, 255, 255);  // BGR white (typical diamond color)
    bool isFilled = true;
    int fillAlpha = 150;              // 0..255 (only used if isFilled)
    int outlineThicknessPx = 2;       // used if !isFilled (or for outline)
};

// Structure to hold a single detected diamond
struct Diamond {
    cv::Point2f center;               // Center point of the diamond
    float radius;                     // Approximate radius
    std::vector<cv::Point> contour;   // Diamond contour points
    cv::Rect boundingBox;             // Bounding box
    int railIndex;                    // Which rail (0=top, 1=right, 2=bottom, 3=left)
    int positionOnRail;               // Position index along the rail (0-based)
};

// Result structure for diamond detection
struct DiamondDetectionResult {
    std::vector<Diamond> diamonds;    // All detected diamonds
    bool ok;                          // True if detection passed validation checks
    int numDiamondsDetected;          // Total number of diamonds detected

    // Per-rail diamond counts (for validation)
    std::array<int, 4> diamondsPerRail;  // [top, right, bottom, left]
};

// Main diamond detection function - works on rectified view with rail masks
// Inputs:
//   rectifiedBgr: The rectified BGR image
//   railMasks: Individual masks for each rail (top, right, bottom, left)
//   params: Detection parameters
DiamondDetectionResult detectDiamonds(
    const cv::Mat& rectifiedBgr,
    const std::vector<cv::Mat>& railMasks,
    const DiamondParams& params
);

// Project diamond detection results from rectified view back to perspective view
// Inputs:
//   rectifiedResult: Diamond detection result in rectified space
//   Hinv: Inverse homography matrix (3x3 CV_64F) to transform from rectified to perspective
//   perspectiveSize: Size of the perspective view image
// Returns:
//   DiamondDetectionResult with all points projected to perspective view
DiamondDetectionResult projectDiamondsToPerspective(
    const DiamondDetectionResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
);

// Draw diamond overlay on an image (in-place modification)
// Draws filled circles or outlines for each detected diamond
void drawDiamondOverlay(cv::Mat& img, const DiamondDetectionResult& result, const DiamondParams& params);

// Draw diamond centers on an image (in-place modification)
// Draws small markers at diamond center points
void drawDiamondCenters(cv::Mat& img, const DiamondDetectionResult& result, const DiamondParams& params);
