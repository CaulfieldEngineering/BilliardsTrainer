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
    
    // Detect felt using a single HSV range (picked-color + sensitivity UI).
    //
    // Hue wrap handling:
    // - If colorHMin <= colorHMax => standard range [HMin..HMax]
    // - Else => wrapped range [0..HMax] U [HMin..180]
    cv::Mat feltMask;
    if (params.colorHMin <= params.colorHMax) {
        cv::inRange(
            hsv,
            cv::Scalar(params.colorHMin, params.colorSMin, params.colorVMin),
            cv::Scalar(params.colorHMax, params.colorSMax, params.colorVMax),
            feltMask
        );
    } else {
        cv::Mat a, b;
        cv::inRange(
            hsv,
            cv::Scalar(0, params.colorSMin, params.colorVMin),
            cv::Scalar(params.colorHMax, params.colorSMax, params.colorVMax),
            a
        );
        cv::inRange(
            hsv,
            cv::Scalar(params.colorHMin, params.colorSMin, params.colorVMin),
            cv::Scalar(180, params.colorSMax, params.colorVMax),
            b
        );
        cv::bitwise_or(a, b, feltMask);
    }
    
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
    
    // Smooth edges by simplifying the contour with polygon approximation
    // This removes jagged edges while maintaining the general direction/shape of each border segment
    std::vector<cv::Point> smoothedContour;
    const double epsilon = 2.5;  // pixels - tune this for more/less smoothing
    cv::approxPolyDP(contours[largestIdx], smoothedContour, epsilon, false);
    
    return smoothedContour;
}

