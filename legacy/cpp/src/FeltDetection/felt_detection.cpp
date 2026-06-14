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

    // FIT 4 STRAIGHT LINES TO EXTREME EDGES OF FELT MASK
    // Strategy: Sample entire width/height, then use median filtering to reject pocket curve outliers

    const int h = feltMask.rows;
    const int w = feltMask.cols;

    std::vector<cv::Point> topEdge, bottomEdge, leftEdge, rightEdge;
    std::vector<int> topY, bottomY, leftX, rightX;

    // TOP EDGE: scan all columns, find topmost white pixel
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (feltMask.at<uchar>(y, x) > 0) {
                topEdge.push_back(cv::Point(x, y));
                topY.push_back(y);
                break;
            }
        }
    }

    // BOTTOM EDGE: scan all columns, find bottommost white pixel
    for (int x = 0; x < w; ++x) {
        for (int y = h - 1; y >= 0; --y) {
            if (feltMask.at<uchar>(y, x) > 0) {
                bottomEdge.push_back(cv::Point(x, y));
                bottomY.push_back(y);
                break;
            }
        }
    }

    // LEFT EDGE: scan all rows, find leftmost white pixel
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (feltMask.at<uchar>(y, x) > 0) {
                leftEdge.push_back(cv::Point(x, y));
                leftX.push_back(x);
                break;
            }
        }
    }

    // RIGHT EDGE: scan all rows, find rightmost white pixel
    for (int y = 0; y < h; ++y) {
        for (int x = w - 1; x >= 0; --x) {
            if (feltMask.at<uchar>(y, x) > 0) {
                rightEdge.push_back(cv::Point(x, y));
                rightX.push_back(x);
                break;
            }
        }
    }

    // Filter outliers using median +/- threshold approach
    // Pocket curves will pull the edge inward, so we filter those out
    auto filterOutliers = [](std::vector<cv::Point>& edge, const std::vector<int>& values, bool keepLarger) {
        if (values.empty()) return;

        // Compute median
        std::vector<int> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        int median = sorted[sorted.size() / 2];

        // Keep only points within reasonable distance of median
        // Pocket curves typically pull 10-30 pixels inward
        const int threshold = 15;

        std::vector<cv::Point> filtered;
        for (size_t i = 0; i < edge.size(); ++i) {
            int val = values[i];
            if (keepLarger) {
                // For right/bottom edges, keep values >= median - threshold
                if (val >= median - threshold) {
                    filtered.push_back(edge[i]);
                }
            } else {
                // For left/top edges, keep values <= median + threshold
                if (val <= median + threshold) {
                    filtered.push_back(edge[i]);
                }
            }
        }
        edge = filtered;
    };

    filterOutliers(topEdge, topY, false);      // Keep smaller y values (closer to top)
    filterOutliers(bottomEdge, bottomY, true); // Keep larger y values (closer to bottom)
    filterOutliers(leftEdge, leftX, false);    // Keep smaller x values (closer to left)
    filterOutliers(rightEdge, rightX, true);   // Keep larger x values (closer to right)

    // Check if we have enough points to fit lines
    if (topEdge.size() < 10 || bottomEdge.size() < 10 ||
        leftEdge.size() < 10 || rightEdge.size() < 10) {
        result.ok = true;
        return result;
    }

    // Fit lines using cv::fitLine with HUBER distance (robust to outliers)
    cv::Vec4f topLine, bottomLine, leftLine, rightLine;
    cv::fitLine(topEdge, topLine, cv::DIST_HUBER, 0, 0.01, 0.01);
    cv::fitLine(bottomEdge, bottomLine, cv::DIST_HUBER, 0, 0.01, 0.01);
    cv::fitLine(leftEdge, leftLine, cv::DIST_HUBER, 0, 0.01, 0.01);
    cv::fitLine(rightEdge, rightLine, cv::DIST_HUBER, 0, 0.01, 0.01);

    // Helper lambda: find intersection of two lines
    // Line format: [vx, vy, x0, y0] where (vx,vy) is direction vector, (x0,y0) is point on line
    auto lineIntersection = [](const cv::Vec4f& L1, const cv::Vec4f& L2) -> cv::Point2f {
        // Line 1: (x,y) = (x1,y1) + t*(vx1,vy1)
        // Line 2: (x,y) = (x2,y2) + s*(vx2,vy2)
        // Solve for intersection
        float vx1 = L1[0], vy1 = L1[1], x1 = L1[2], y1 = L1[3];
        float vx2 = L2[0], vy2 = L2[1], x2 = L2[2], y2 = L2[3];

        float det = vx1 * vy2 - vy1 * vx2;
        if (std::abs(det) < 1e-6) {
            // Lines are parallel, return midpoint
            return cv::Point2f((x1 + x2) / 2, (y1 + y2) / 2);
        }

        float dx = x2 - x1;
        float dy = y2 - y1;
        float t = (dx * vy2 - dy * vx2) / det;

        return cv::Point2f(x1 + t * vx1, y1 + t * vy1);
    };

    // Compute 4 corner intersections
    cv::Point2f topLeft = lineIntersection(topLine, leftLine);
    cv::Point2f topRight = lineIntersection(topLine, rightLine);
    cv::Point2f bottomRight = lineIntersection(bottomLine, rightLine);
    cv::Point2f bottomLeft = lineIntersection(bottomLine, leftLine);

    result.corners = {topLeft, topRight, bottomRight, bottomLeft};
    result.hasCorners = true;

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

    // Draw quad if corners are available (using default green color for debug view)
    if (result.hasCorners) {
        std::vector<cv::Point> quadPts;
        for (const auto& corner : result.corners) {
            quadPts.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
        }
        cv::polylines(debug, std::vector<std::vector<cv::Point>>{quadPts}, true, cv::Scalar(0, 255, 255), 2);
    }

    return debug;
}

// Draw felt overlay on an image (in-place modification)
// Applies the felt contour overlay with styling from params
void drawFeltOverlay(cv::Mat& img, const FeltDetectionResult& result, const FeltParams& params) {
    if (result.contour.empty()) {
        return;
    }
    
    std::vector<std::vector<cv::Point>> contours{result.contour};
    
    if (params.isFilled) {
        cv::Mat overlay = img.clone();
        cv::fillPoly(overlay, contours, params.color);
        const double a = std::clamp(params.fillAlpha, 0, 255) / 255.0;
        cv::addWeighted(img, 1.0 - a, overlay, a, 0, img);
        // Outline for definition
        cv::drawContours(img, contours, -1, params.color, std::max(1, params.outlineThicknessPx));
    } else {
        cv::drawContours(img, contours, -1, params.color, std::max(1, params.outlineThicknessPx));
    }
}

// Draw felt overlay from a mask (in-place modification)
// Applies the felt mask overlay with styling from params
void drawFeltOverlayFromMask(cv::Mat& img, const cv::Mat& feltMask, const FeltParams& params) {
    if (feltMask.empty()) {
        return;
    }
    
    if (params.isFilled) {
        cv::Mat overlay = img.clone();
        // Create a 3-channel mask for blending
        std::vector<cv::Mat> maskChannels;
        cv::split(feltMask, maskChannels);
        cv::Mat mask3ch;
        cv::merge(std::vector<cv::Mat>{maskChannels[0], maskChannels[0], maskChannels[0]}, mask3ch);
        
        // Apply color overlay where mask is non-zero
        overlay.setTo(params.color, mask3ch);
        const double a = std::clamp(params.fillAlpha, 0, 255) / 255.0;
        cv::addWeighted(img, 1.0 - a, overlay, a, 0, img);
        
        // Draw outline for definition
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(feltMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            cv::drawContours(img, contours, -1, params.color, std::max(1, params.outlineThicknessPx));
        }
    } else {
        // Draw outline only
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(feltMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            cv::drawContours(img, contours, -1, params.color, std::max(1, params.outlineThicknessPx));
        }
    }
}

// Draw felt corners quad on an image (in-place modification)
// This draws the quad that hugs the felt table corners (tightest path around felt)
// Uses a darker version of the felt overlay color
void drawFeltCornersQuad(cv::Mat& img, const FeltDetectionResult& result, const FeltParams& params) {
    if (result.hasCorners) {
        std::vector<cv::Point> quadPts;
        for (const auto& corner : result.corners) {
            quadPts.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
        }
        
        // Use a darker version of the felt overlay color (multiply each channel by 0.7)
        cv::Scalar darkerColor(
            params.color[0] * 0.7,
            params.color[1] * 0.7,
            params.color[2] * 0.7
        );
        
        cv::polylines(img, std::vector<std::vector<cv::Point>>{quadPts}, true, darkerColor, 2);
    }
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
