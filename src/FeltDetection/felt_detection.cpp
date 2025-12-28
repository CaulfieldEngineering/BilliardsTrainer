#include "felt_detection.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

// Helper function: Keep only the largest connected component in a mask
static void keepLargestComponent(cv::Mat& mask) {
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    
    if (nLabels <= 1) {
        // Only background, nothing to keep
        mask.setTo(0);
        return;
    }
    
    // Find the component with maximum area (excluding background at index 0)
    int maxAreaIdx = 1;
    int maxArea = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < nLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }
    
    // Create new mask with only the largest component
    cv::Mat result = cv::Mat::zeros(mask.size(), mask.type());
    result.setTo(255, labels == maxAreaIdx);
    mask = result;
}

// Helper function: Fill holes in mask using flood fill from border
// Standard flood-fill hole-fill pattern: flood fill outside, then holes = inverted regions not connected to border
static void fillHoles(cv::Mat& mask) {
    if (mask.empty()) return;
    
    // Ensure binary 0/255
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    
    // Invert: holes become white, background becomes white, felt becomes black
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    
    // Flood fill the OUTSIDE background in the inverted image
    // Use a padded image so floodFill doesn't have edge issues
    cv::Mat ff = inv.clone();
    cv::floodFill(ff, cv::Point(0, 0), 255);
    
    // Now: ff has outside filled to 255.
    // The holes are the remaining white regions in inv that were NOT connected to the border.
    // holes = inv AND (NOT ff)
    cv::Mat ff_not, holes;
    cv::bitwise_not(ff, ff_not);
    cv::bitwise_and(inv, ff_not, holes);
    
    // Fill holes into original mask
    cv::bitwise_or(mask, holes, mask);
}

// Helper function: Order 4 corners as TL, TR, BR, BL
// Robust ordering: sort by y (top two first), then by x within top/bottom pairs
static void orderCorners(std::array<cv::Point2f, 4>& c) {
    // Sort by y (top two first)
    std::sort(c.begin(), c.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    
    // Top two are c[0], c[1]; bottom two are c[2], c[3]
    // Within each pair, sort by x to get left/right
    cv::Point2f tl = (c[0].x < c[1].x) ? c[0] : c[1];
    cv::Point2f tr = (c[0].x < c[1].x) ? c[1] : c[0];
    cv::Point2f bl = (c[2].x < c[3].x) ? c[2] : c[3];
    cv::Point2f br = (c[2].x < c[3].x) ? c[3] : c[2];
    
    c = {tl, tr, br, bl};
}

// Main felt detection function - returns complete result structure
FeltDetectionResult detectFelt(const cv::Mat& bgr, const FeltParams& params) {
    FeltDetectionResult result;
    result.ok = false;
    
    if (bgr.empty()) {
        return result;
    }
    
    // Convert to HSV
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    
    // Detect felt using HSV range (with hue wrap handling)
    cv::Mat feltMask;
    if (params.colorHMin <= params.colorHMax) {
        cv::inRange(
            hsv,
            cv::Scalar(params.colorHMin, params.colorSMin, params.colorVMin),
            cv::Scalar(params.colorHMax, params.colorSMax, params.colorVMax),
            feltMask
        );
    } else {
        // Wrapped range: [0..HMax] U [HMin..180]
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
    
    // Ensure binary 0/255 before connected components
    cv::threshold(feltMask, feltMask, 127, 255, cv::THRESH_BINARY);
    
    // MASK HARDENING STEP 1: Keep only largest connected component
    keepLargestComponent(feltMask);
    
    // MASK HARDENING STEP 2: Fill holes
    fillHoles(feltMask);
    
    // Sanity check: if mask is nearly all white, something went wrong
    double white = cv::countNonZero(feltMask);
    double ratio = white / (feltMask.rows * feltMask.cols);
    if (ratio > 0.95) {
        // Something went wrong; likely bad thresholds or fill logic
        // Return empty result to indicate failure
        result.feltMask = cv::Mat::zeros(feltMask.size(), feltMask.type());
        return result;
    }
    
    // Find external contour
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(feltMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        result.feltMask = feltMask;
        return result;
    }
    
    // Get the largest contour (should be only one after mask hardening, but be safe)
    size_t largestIdx = 0;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIdx = i;
        }
    }
    
    result.contour = contours[largestIdx];
    result.feltMask = feltMask;
    result.hasCorners = false;
    
    // Geometric normalization: compute convex hull to remove pocket indentations
    // This restores the true rectangular shape and prevents corners from being pulled inward by pockets
    std::vector<cv::Point> hull;
    cv::convexHull(result.contour, hull);
    
    // Extract 4-corner quadrilateral from the convex hull
    double perim = cv::arcLength(hull, true);
    std::vector<cv::Point> poly;
    
    // Try epsilon sweep to get exactly 4 points
    const double epsFracs[] = {0.005, 0.01, 0.015, 0.02, 0.03};
    bool found4Points = false;
    for (double epsFrac : epsFracs) {
        double eps = epsFrac * perim;
        cv::approxPolyDP(hull, poly, eps, true);
        if (poly.size() == 4) {
            found4Points = true;
            break;
        }
    }
    
    // Fallback: use minimum area rectangle if we couldn't get 4 points
    if (!found4Points) {
        cv::RotatedRect minRect = cv::minAreaRect(hull);
        cv::Point2f boxPoints[4];
        minRect.points(boxPoints);
        poly.clear();
        for (int i = 0; i < 4; ++i) {
            poly.push_back(cv::Point(static_cast<int>(boxPoints[i].x), static_cast<int>(boxPoints[i].y)));
        }
    }
    
    // Store raw poly for debug visualization
    result.polyDebug = poly;
    
    // Convert to Point2f and order as TL, TR, BR, BL
    // After the above, we should always have at least 4 points (either from approxPolyDP or minAreaRect fallback)
    if (poly.size() >= 4) {
        for (int i = 0; i < 4; ++i) {
            result.corners[i] = cv::Point2f(static_cast<float>(poly[i].x), static_cast<float>(poly[i].y));
        }
        orderCorners(result.corners);
        result.hasCorners = true;
    }
    
    // Store telemetry fields
    result.polySize = static_cast<int>(poly.size());
    result.found4Points = found4Points;
    
    // Debug prints to verify corner computation
    std::cout << "hull size=" << hull.size()
              << " poly size=" << poly.size()
              << " found4=" << found4Points
              << " hasCorners=" << result.hasCorners
              << std::endl;
    
    // VALIDATION GUARDRAILS - use convex hull envelope for all geometry checks
    double envArea = cv::contourArea(hull);
    cv::Rect envBox = cv::boundingRect(hull);
    result.bbox = envBox; // Use envelope bbox
    
    double imageArea = static_cast<double>(bgr.cols * bgr.rows);
    double areaRatio = envArea / imageArea;
    
    // Store telemetry fields
    result.envArea = envArea;
    result.areaRatio = areaRatio;
    
    // Area ratio check: should cover significant portion of image
    if (areaRatio < 0.05) {
        return result; // ok = false
    }
    
    // Aspect ratio check: plausible for pool table (wide or tall)
    double aspectRatio = static_cast<double>(envBox.width) / static_cast<double>(envBox.height);
    if (aspectRatio < 0.8 || aspectRatio > 3.0) {
        return result; // ok = false
    }
    
    // Solidity check removed: hull/hull = 1.0, so not meaningful
    // The hull is our envelope, so we validate based on it directly
    
    // All checks passed
    result.ok = true;
    return result;
}

// Debug visualization function
cv::Mat drawFeltDebug(const cv::Mat& bgr, const FeltDetectionResult& result) {
    cv::Mat debug = bgr.clone();
    
    if (result.feltMask.empty() || result.contour.empty()) {
        return debug;
    }
    
    // Draw mask overlay (alpha blend)
    cv::Mat maskOverlay = debug.clone();
    maskOverlay.setTo(cv::Scalar(0, 255, 0), result.feltMask);
    cv::addWeighted(debug, 0.7, maskOverlay, 0.3, 0, debug);
    
    // Draw contour in green
    std::vector<std::vector<cv::Point>> contours = {result.contour};
    cv::drawContours(debug, contours, -1, cv::Scalar(0, 255, 0), 2);
    
    // Draw hull (cyan) so you can see the envelope you're using
    std::vector<cv::Point> hull;
    cv::convexHull(result.contour, hull);
    if (!hull.empty()) {
        std::vector<std::vector<cv::Point>> hullContours = {hull};
        cv::polylines(debug, hull, true, cv::Scalar(255, 255, 0), 2); // cyan
    }
    
    // Draw raw poly (magenta) regardless of hasCorners flag
    if (!result.polyDebug.empty()) {
        cv::polylines(debug, result.polyDebug, true, cv::Scalar(255, 0, 255), 2);
        for (size_t i = 0; i < result.polyDebug.size(); ++i) {
            cv::circle(debug, result.polyDebug[i], 8, cv::Scalar(255, 0, 255), 2);
        }
    }
    
    // Draw corners as numbered circles and lines connecting them
    // Draw whenever corners were successfully computed, independent of validation (ok)
    if (result.hasCorners) {
        const cv::Scalar cornerColor(255, 0, 255); // Magenta
        const int cornerRadius = 8;
        const int cornerThickness = 2;
        
        // Draw corner points with numbers
        for (size_t i = 0; i < 4; ++i) {
            cv::Point p(static_cast<int>(result.corners[i].x), static_cast<int>(result.corners[i].y));
            cv::circle(debug, p, cornerRadius, cornerColor, cornerThickness);
            
            // Draw number label
            std::string label = std::to_string(i);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::putText(debug, label, cv::Point(p.x - textSize.width / 2, p.y - cornerRadius - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cornerColor, 2);
        }
        
        // Draw lines connecting corners
        for (int i = 0; i < 4; ++i) {
            cv::Point p1(static_cast<int>(result.corners[i].x), static_cast<int>(result.corners[i].y));
            cv::Point p2(static_cast<int>(result.corners[(i + 1) % 4].x), 
                        static_cast<int>(result.corners[(i + 1) % 4].y));
            cv::line(debug, p1, p2, cornerColor, 2);
        }
    }
    
    // Draw bounding box
    cv::rectangle(debug, result.bbox, cv::Scalar(255, 255, 0), 2);
    
    return debug;
}

// Wrapper functions for backward compatibility

cv::Rect detectFeltArea(const cv::Mat& src) {
    return detectFeltArea(src, FeltParams{});
}

cv::Rect detectFeltArea(const cv::Mat& src, const FeltParams& params) {
    FeltDetectionResult result = detectFelt(src, params);
    if (!result.ok || result.contour.empty()) {
        return cv::Rect(0, 0, src.cols, src.rows);
    }
    return result.bbox;
}

std::vector<cv::Point> detectFeltContour(const cv::Mat& src) {
    return detectFeltContour(src, FeltParams{});
}

std::vector<cv::Point> detectFeltContour(const cv::Mat& src, const FeltParams& params) {
    FeltDetectionResult result = detectFelt(src, params);
    return result.contour; // Returns empty vector if !ok
}
