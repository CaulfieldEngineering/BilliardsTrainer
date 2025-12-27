#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

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

// Detect the felt/table surface (blue or green)
// Returns the bounding rectangle
cv::Rect detectFeltArea(const cv::Mat& src);
cv::Rect detectFeltArea(const cv::Mat& src, const FeltParams& params);

// Detect the felt/table surface and return the actual contour
// Returns the largest contour found (the felt perimeter)
std::vector<cv::Point> detectFeltContour(const cv::Mat& src);
std::vector<cv::Point> detectFeltContour(const cv::Mat& src, const FeltParams& params);

