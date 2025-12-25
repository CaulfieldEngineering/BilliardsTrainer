#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

// Group diamonds by rail (top, bottom, left, right)
struct RailDiamonds {
    std::vector<cv::Point> top;      // Short rail - 3 diamonds
    std::vector<cv::Point> bottom;   // Short rail - 3 diamonds
    std::vector<cv::Point> left;     // Long rail - 6 diamonds (3+3)
    std::vector<cv::Point> right;    // Long rail - 6 diamonds (3+3)
};

RailDiamonds groupDiamondsByRail(const std::vector<cv::Point>& diamonds, const cv::Rect& feltRect) {
    RailDiamonds railDiamonds;
    
    // Calculate rail boundaries (extend felt rect outward)
    int railWidth = std::max(feltRect.width, feltRect.height) / 15; // Approximate rail width
    
    int topBoundary = feltRect.y - railWidth;
    int bottomBoundary = feltRect.y + feltRect.height + railWidth;
    int leftBoundary = feltRect.x - railWidth;
    int rightBoundary = feltRect.x + feltRect.width + railWidth;
    
    int centerX = feltRect.x + feltRect.width / 2;
    int centerY = feltRect.y + feltRect.height / 2;
    
    for (const auto& pt : diamonds) {
        // Determine which rail based on position relative to felt
        if (pt.y < topBoundary && pt.y < centerY) {
            railDiamonds.top.push_back(pt);
        } else if (pt.y > bottomBoundary && pt.y > centerY) {
            railDiamonds.bottom.push_back(pt);
        } else if (pt.x < leftBoundary && pt.x < centerX) {
            railDiamonds.left.push_back(pt);
        } else if (pt.x > rightBoundary && pt.x > centerX) {
            railDiamonds.right.push_back(pt);
        }
    }
    
    return railDiamonds;
}

// Validate and filter diamonds based on geometric constraints
std::vector<cv::Point> validateDiamondGrid(const std::vector<cv::Point>& diamonds, int expectedCount, bool isLongRail) {
    if (diamonds.size() <= expectedCount) {
        return diamonds; // If we have the right number or fewer, return as-is
    }
    
    // Sort diamonds along the rail
    std::vector<cv::Point> sorted = diamonds;
    if (isLongRail) {
        // Sort by Y coordinate for long rails (left/right)
        std::sort(sorted.begin(), sorted.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        });
        
        // For long rails, we expect 6 diamonds split into 2 groups of 3
        // Find the gap (pocket location) and take 3 from each side
        if (sorted.size() >= 6) {
            // Find largest gap (where pocket is)
            int maxGap = 0;
            int gapIdx = 0;
            for (size_t i = 1; i < sorted.size(); ++i) {
                int gap = sorted[i].y - sorted[i-1].y;
                if (gap > maxGap) {
                    maxGap = gap;
                    gapIdx = i;
                }
            }
            
            // Take 3 from before gap and 3 from after gap
            std::vector<cv::Point> result;
            int startIdx = std::max(0, static_cast<int>(gapIdx) - 3);
            int endIdx = std::min(static_cast<int>(sorted.size()), startIdx + 6);
            for (int i = startIdx; i < endIdx; ++i) {
                result.push_back(sorted[i]);
            }
            return result;
        }
    } else {
        // Sort by X coordinate for short rails (top/bottom)
        std::sort(sorted.begin(), sorted.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.x < b.x;
        });
        
        // For short rails, take the 3 most evenly spaced
        if (sorted.size() >= 3) {
            // Calculate spacing and take the 3 that form the most regular grid
            std::vector<cv::Point> result;
            int step = sorted.size() / 3;
            for (int i = 0; i < 3; ++i) {
                result.push_back(sorted[i * step]);
            }
            return result;
        }
    }
    
    return sorted;
}

// Detect diamond markers on the pool table rails with context-aware constraints
void detectDiamonds(const cv::Mat& src, cv::Mat& dst, bool showDiamonds, const DiamondDetectionParams& params) {
    if (!showDiamonds) {
        return;
    }
    
    // Step 1: Detect the felt/table surface to understand the table boundaries
    std::vector<cv::Point> feltContour = detectFeltContour(src, FeltParams{});
    cv::Rect feltRect = feltContour.empty() ? cv::Rect(0, 0, src.cols, src.rows) 
                                            : cv::boundingRect(feltContour);
    
    // Step 2: Detect dark rail areas (where diamonds should be) - adjacent to felt perimeter
    cv::Mat railMask = detectRailMask(src, feltContour, RailParams{});
    
    // Step 3: Detect white/light circular markers in the rail areas
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Threshold for white/light colors (diamonds are typically white)
    // Relaxed thresholds to catch more candidates
    cv::Mat whiteMask;
    cv::inRange(hsv, cv::Scalar(0, 0, 180), cv::Scalar(180, 50, 255), whiteMask);
    
    // Only look for white markers in rail areas
    cv::Mat candidateMask;
    cv::bitwise_and(whiteMask, railMask, candidateMask);
    
    // If rail mask is empty or too small, try without rail constraint as fallback
    int railPixels = cv::countNonZero(railMask);
    if (railPixels < 100) {
        // Rail detection might have failed, use white mask directly but still filter by position
        candidateMask = whiteMask;
    }
    
    // Apply morphological operations to clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(candidateMask, candidateMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(candidateMask, candidateMask, cv::MORPH_OPEN, kernel);
    
    // Find contours of potential diamonds
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(candidateMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<cv::Point> diamondCenters;
    
    // Filter contours to find circular markers
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < params.minArea || area > params.maxArea) {
            continue;
        }
        
        // Check circularity
        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;
        
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        // Accept reasonably circular shapes
        if (circularity > 0.5 && circularity < 1.3) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 != 0) {
                cv::Point center(static_cast<int>(m.m10 / m.m00),
                                static_cast<int>(m.m01 / m.m00));
                
                // Verify the marker is on a dark background (rail)
                // Check surrounding area
                int checkRadius = 15;
                cv::Rect checkRect(
                    std::max(0, center.x - checkRadius),
                    std::max(0, center.y - checkRadius),
                    std::min(src.cols - center.x + checkRadius, 2 * checkRadius),
                    std::min(src.rows - center.y + checkRadius, 2 * checkRadius)
                );
                
                cv::Mat checkRoi = hsv(checkRect);
                cv::Scalar meanVal = cv::mean(checkRoi);
                
                // Background should be dark (low value in HSV)
                // Relaxed threshold - allow slightly brighter backgrounds
                if (meanVal[2] < 120) { // Dark background (relaxed from 100)
                    diamondCenters.push_back(center);
                }
            }
        }
    }
    
    // Step 4: Group diamonds by rail and apply geometric constraints
    RailDiamonds railDiamonds = groupDiamondsByRail(diamondCenters, feltRect);
    
    // Step 5: Validate and filter based on expected counts
    railDiamonds.top = validateDiamondGrid(railDiamonds.top, 3, false);
    railDiamonds.bottom = validateDiamondGrid(railDiamonds.bottom, 3, false);
    railDiamonds.left = validateDiamondGrid(railDiamonds.left, 6, true);
    railDiamonds.right = validateDiamondGrid(railDiamonds.right, 6, true);
    
    // Step 6: Draw validated diamonds
    std::vector<cv::Point> allValidatedDiamonds;
    allValidatedDiamonds.insert(allValidatedDiamonds.end(), railDiamonds.top.begin(), railDiamonds.top.end());
    allValidatedDiamonds.insert(allValidatedDiamonds.end(), railDiamonds.bottom.begin(), railDiamonds.bottom.end());
    allValidatedDiamonds.insert(allValidatedDiamonds.end(), railDiamonds.left.begin(), railDiamonds.left.end());
    allValidatedDiamonds.insert(allValidatedDiamonds.end(), railDiamonds.right.begin(), railDiamonds.right.end());
    
    const int radius = std::max(1, params.radiusPx);
    const int outline = std::max(1, params.outlineThicknessPx);

    const double a = std::clamp(params.alpha, 0, 255) / 255.0;
    if (a <= 0.0 || allValidatedDiamonds.empty()) {
        return;
    }

    // If alpha is fully opaque, draw directly onto dst (fast path).
    if (a >= 1.0) {
        for (const auto& center : allValidatedDiamonds) {
            if (params.isFilled) {
                cv::circle(dst, center, radius, params.color, -1);
                cv::circle(dst, center, radius, params.color, outline);
            } else {
                cv::circle(dst, center, radius, params.color, outline);
            }
        }
        return;
    }

    // Otherwise draw onto an overlay and blend once.
    cv::Mat overlay = dst.clone();
    for (const auto& center : allValidatedDiamonds) {
        if (params.isFilled) {
            cv::circle(overlay, center, radius, params.color, -1);
            cv::circle(overlay, center, radius, params.color, outline);
        } else {
            cv::circle(overlay, center, radius, params.color, outline);
        }
    }
    cv::addWeighted(dst, 1.0 - a, overlay, a, 0, dst);
}

