#include "felt_detection.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

// Detect the felt/table surface (blue or green)
cv::Rect detectFeltArea(const cv::Mat& src) {
    return detectFeltArea(src, FeltParams{});
}

cv::Rect detectFeltArea(const cv::Mat& src, const FeltParams& params) {
    std::vector<cv::Point> feltContour = detectFeltContour(src, params);
    if (feltContour.empty()) {
        return cv::Rect(0, 0, src.cols, src.rows);
    }
    return cv::boundingRect(feltContour);
}

// Detect the felt/table surface and return the actual contour
std::vector<cv::Point> detectFeltContour(const cv::Mat& src) {
    return detectFeltContour(src, FeltParams{});
}

std::vector<cv::Point> detectFeltContour(const cv::Mat& src, const FeltParams& params) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Detect blue felt (common pool table color)
    cv::Mat blueMask;
    cv::inRange(
        hsv,
        cv::Scalar(params.blueHMin, params.blueSMin, params.blueVMin),
        cv::Scalar(params.blueHMax, params.blueSMax, params.blueVMax),
        blueMask
    );
    
    // Also detect green felt
    cv::Mat greenMask;
    cv::inRange(
        hsv,
        cv::Scalar(params.greenHMin, params.greenSMin, params.greenVMin),
        cv::Scalar(params.greenHMax, params.greenSMax, params.greenVMax),
        greenMask
    );
    
    // Combine masks
    cv::Mat feltMask;
    cv::bitwise_or(blueMask, greenMask, feltMask);
    
    // Apply morphological operations to clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(feltMask, feltMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(feltMask, feltMask, cv::MORPH_OPEN, kernel);
    
    // Find the largest contour (the felt)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(feltMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return std::vector<cv::Point>(); // Return empty if no felt detected
    }
    
    // Find largest contour
    size_t largestIdx = 0;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIdx = i;
        }
    }
    
    return contours[largestIdx];
}

