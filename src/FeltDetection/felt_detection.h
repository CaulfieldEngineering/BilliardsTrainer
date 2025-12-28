#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

// Structure to hold felt detection parameters
struct FeltParams {
    // Felt color filtering (HSV range)
    //
    // The old model exposed separate "blue felt" and "green felt" HSV slider ranges.
    // We now use the same UI concept as Diamonds:
    // - user picks a representative felt color from the live frame
    // - a Sensitivity slider expands/contracts the accepted HSV range around that picked color
    //
    // Hue wrapping is represented by allowing colorHMin > colorHMax (range spans 180->0).
    int colorHMin = 40, colorHMax = 80;   // default: "green-ish"
    int colorSMin = 50, colorSMax = 255;
    int colorVMin = 50, colorVMax = 255;

    // Color picker state (UI convenience)
    bool hasPickedColor = true;
    // Defaults chosen to represent typical green felt in OpenCV HSV space.
    // (H: 0..180, S/V: 0..255)
    cv::Vec3b pickedHSV = cv::Vec3b(60, 200, 200);
    cv::Vec3b pickedBGR = cv::Vec3b(0, 255, 0);
    // Default tuned from practical use: 82 tends to handle rail shadows without bleeding too far.
    int colorSensitivity = 82; // 0..100 (strict..loose)

    // Overlay styling
    cv::Scalar color = cv::Scalar(0, 255, 0); // BGR
    bool isFilled = true;
    int fillAlpha = 80;          // 0..255 (only used if isFilled)
    int outlineThicknessPx = 2;  // used if !isFilled (or for outline)
};

// Result structure for robust felt detection
// Provides clean mask, stable contour, and ordered 4-corner quad suitable for rail extraction
struct FeltDetectionResult {
    cv::Mat feltMask;                              // CV_8U, 0/255 - cleaned mask with holes filled
    std::vector<cv::Point> contour;                // External contour of felt (raw, before simplification)
    std::array<cv::Point2f, 4> corners;            // Ordered corners: TL, TR, BR, BL
    cv::Rect bbox;                                 // Bounding rectangle of convex hull envelope
    bool ok;                                       // True if detection passed validation checks
    bool hasCorners;                               // True if 4 corners were successfully computed
    std::vector<cv::Point> polyDebug;              // Temporary: raw polygon for debug visualization
    
    // Telemetry fields for export
    double envArea = 0.0;                          // Area of convex hull envelope
    double areaRatio = 0.0;                        // envArea / imageArea
    int polySize = 0;                              // Size of poly used to form corners
    bool found4Points = false;                     // True if approxPolyDP produced exactly 4 points
};

// Main felt detection function - returns complete result structure
FeltDetectionResult detectFelt(const cv::Mat& bgr, const FeltParams& params);

// Debug visualization function - draws mask overlay, contour, corners, and bbox
cv::Mat drawFeltDebug(const cv::Mat& bgr, const FeltDetectionResult& result);

// Draw felt overlay on an image (in-place modification)
// Applies the felt contour overlay with styling from params
void drawFeltOverlay(cv::Mat& img, const FeltDetectionResult& result, const FeltParams& params);

// Draw felt overlay from a mask (in-place modification)
// Applies the felt mask overlay with styling from params
void drawFeltOverlayFromMask(cv::Mat& img, const cv::Mat& feltMask, const FeltParams& params);

// Draw felt corners quad on an image (in-place modification)
// This draws the quad that hugs the felt table corners (tightest path around felt)
// Uses a darker version of the felt overlay color
void drawFeltCornersQuad(cv::Mat& img, const FeltDetectionResult& result, const FeltParams& params);

// Detect the felt/table surface (blue or green)
// Returns the bounding rectangle
cv::Rect detectFeltArea(const cv::Mat& src);
cv::Rect detectFeltArea(const cv::Mat& src, const FeltParams& params);

// Detect the felt/table surface and return the actual contour
// Returns the largest contour found (the felt perimeter)
std::vector<cv::Point> detectFeltContour(const cv::Mat& src);
std::vector<cv::Point> detectFeltContour(const cv::Mat& src, const FeltParams& params);

