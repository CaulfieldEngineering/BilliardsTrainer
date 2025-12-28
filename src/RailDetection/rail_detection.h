#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Structure to hold rail detection parameters
struct RailParams {
    // Black rail HSV range
    // Default increased to better match typical "dark rail" brightness under indoor lighting.
    int blackVMax = 217;
    // Brown rail HSV range
    int brownHMax = 40;
    int brownSMax = 120;
    int brownVMax = 130;
    // Overlay styling
    cv::Scalar color = cv::Scalar(0, 165, 255); // BGR
    bool isFilled = true;
    int fillAlpha = 80;          // 0..255 (only used if isFilled)
    int outlineThicknessPx = 2;
};

// Detect the dark rail surface around the felt (where diamonds are printed)
// Rails are always adjacent to the felt perimeter
// Returns a mask of the rail areas
cv::Mat detectRailMask(const cv::Mat& src, const std::vector<cv::Point>& feltContour);
cv::Mat detectRailMask(const cv::Mat& src, const std::vector<cv::Point>& feltContour, const RailParams& params);

// Detect rail areas and return contours for visualization
std::vector<std::vector<cv::Point>> detectRailContours(const cv::Mat& src, const std::vector<cv::Point>& feltContour);
std::vector<std::vector<cv::Point>> detectRailContours(const cv::Mat& src, const std::vector<cv::Point>& feltContour, const RailParams& params);

// Extract a straight, 4-corner outer boundary for the rail area from an already-computed rail mask.
// This is a *spatial* straightening step (no temporal averaging): it replaces jagged edges with straight lines.
//
// Returns an empty vector if a boundary could not be estimated.
std::vector<cv::Point> detectRailOuterBoundary(const cv::Mat& railMask);

// Structure to hold the four edge lines of the rail mask
// Each line is represented by two points (pt1, pt2) that extend to the image boundaries
struct RailEdgeLines {
    cv::Point topLinePt1;      // Left endpoint of top edge line
    cv::Point topLinePt2;      // Right endpoint of top edge line
    cv::Point bottomLinePt1;   // Left endpoint of bottom edge line
    cv::Point bottomLinePt2;   // Right endpoint of bottom edge line
    cv::Point leftLinePt1;     // Top endpoint of left edge line
    cv::Point leftLinePt2;     // Bottom endpoint of left edge line
    cv::Point rightLinePt1;    // Top endpoint of right edge line
    cv::Point rightLinePt2;    // Bottom endpoint of right edge line
};

// Fit four lines to the edges of the rail mask (top, bottom, left, right)
// Each line extends to infinity (image boundaries) and represents a best fit to the corresponding edge
// Returns true if all four lines were successfully fitted, false otherwise
bool detectRailEdgeLines(const cv::Mat& railMask, const cv::Size& imageSize, RailEdgeLines& edgeLines);

