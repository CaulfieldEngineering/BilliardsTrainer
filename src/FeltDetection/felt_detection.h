#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Structure to hold felt detection parameters
struct FeltParams {
    // Blue felt HSV range
    int blueHMin = 100, blueHMax = 130;
    int blueSMin = 50, blueSMax = 255;
    int blueVMin = 50, blueVMax = 255;
    // Green felt HSV range
    int greenHMin = 40, greenHMax = 80;
    int greenSMin = 50, greenSMax = 255;
    int greenVMin = 50, greenVMax = 255;
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

