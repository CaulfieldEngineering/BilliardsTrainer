#include "rail_detection.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <queue>

// ---------------------------------------------------------------------------------------------
// Rail boundary straightening helpers
// ---------------------------------------------------------------------------------------------
static std::vector<cv::Point> computeOuterBoundaryQuadFromMask(const cv::Mat& railMask) {
    if (railMask.empty() || railMask.type() != CV_8UC1) return {};

    // Find contours on the mask and take the largest as the rail region.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(railMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    size_t largestIdx = 0;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > largestArea) {
            largestArea = a;
            largestIdx = i;
        }
    }
    const auto& c = contours[largestIdx];
    if (c.size() < 10) return {};

    // First try a polygon approximation to get a 4-corner shape.
    std::vector<cv::Point> approx;
    const double peri = cv::arcLength(c, true);
    // Epsilon chosen to aggressively straighten (reduce jaggedness) while still matching the table shape.
    cv::approxPolyDP(c, approx, 0.02 * peri, true);

    if (approx.size() == 4) {
        // Ensure convex ordering
        std::vector<cv::Point> hull;
        cv::convexHull(approx, hull);
        if (hull.size() == 4) return hull;
        return approx;
    }

    // Fallback: min-area rectangle is robust even when corners are partially missing.
    cv::RotatedRect rr = cv::minAreaRect(c);
    cv::Point2f pts2f[4];
    rr.points(pts2f);
    std::vector<cv::Point> pts;
    pts.reserve(4);
    for (int i = 0; i < 4; ++i) {
        pts.emplace_back(cv::Point((int)std::lround(pts2f[i].x), (int)std::lround(pts2f[i].y)));
    }
    return pts;
}

// Helper function to check if a pixel is dark (black/brown rail color)
// Uses thresholds from params to account for lighting variations.
static bool isDarkPixel(const cv::Vec3b& hsvPixel, const RailParams& params) {
    int h = hsvPixel[0];
    int s = hsvPixel[1];
    int v = hsvPixel[2];
    
    // Check for black/dark gray (increased threshold to account for lighting)
    // Rails can appear brighter due to lighting, so we allow up to V=params.blackVMax
    if (v < params.blackVMax) {
        return true;
    }
    
    // Check for dark brown/wood (low value, low-medium saturation)
    if (v < params.brownVMax && s < params.brownSMax && h < params.brownHMax) {
        return true;
    }
    
    return false;
}

// Detect the rail mask as a narrow band *outside* the felt boundary.
//
// Goals / constraints:
// - **Exclude felt** entirely (mask must not include felt pixels).
// - **Halt at the felt/rail boundary** (we never grow into the felt region).
// - **Prevent bleeding** into background objects (e.g. top-left clutter) and into the table body/legs.
//
// Approach:
// - Build a binary felt mask from the felt contour.
// - Use a distance transform on the non-felt area to create a "ring" (band) within N pixels of the felt.
// - Intersect that ring with a "dark rail color" mask (black/brown in HSV).
// - Remove small connected components (noise/bleed candidates).
cv::Mat detectRailMask(const cv::Mat& src, const std::vector<cv::Point>& feltContour) {
    return detectRailMask(src, feltContour, RailParams{});
}

cv::Mat detectRailMask(const cv::Mat& src, const std::vector<cv::Point>& feltContour, const RailParams& params) {
    if (feltContour.empty()) {
        return cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    // Step 1: Create a mask of the felt area (to exclude from rail detection)
    cv::Mat feltMask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> feltContours;
    feltContours.push_back(feltContour);
    cv::fillPoly(feltMask, feltContours, cv::Scalar(255));

    // IMPORTANT:
    // The felt contour can be slightly conservative (under-fit) depending on thresholds/lighting,
    // which can leave a thin strip of actual felt outside `feltMask`.
    //
    // To ensure the rail mask never includes felt, we expand the felt mask slightly and use that
    // expanded version for all "outside felt" constraints and for a final subtraction pass.
    cv::Mat feltMaskExpanded;
    {
        const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::dilate(feltMask, feltMaskExpanded, k);
    }
    
    // Step 2: Convert to HSV for color-based detection
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Step 3: Build a "dark pixel" mask for rail colors (black/brown in HSV).
    // This is the vectorized equivalent of `isDarkPixel()`.
    cv::Mat blackMask;
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, params.blackVMax), blackMask);

    cv::Mat brownMask;
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(params.brownHMax, params.brownSMax, params.brownVMax), brownMask);

    cv::Mat darkMask;
    cv::bitwise_or(blackMask, brownMask, darkMask);

    // Step 4: Compute distance-to-felt (outside only) so we can:
    // - seed *right at* the felt/rail boundary
    // - then expand outward to the rail's *outer edge* while still being bounded.
    cv::Rect feltRect = cv::boundingRect(feltContour);
    // Estimate rail thickness in pixels as a function of the felt size.
    // We intentionally bias larger than before so the mask reaches the *outer* rail edge,
    // but we still bound it so we don't leak into the table body/legs.
    const int minDim = std::max(1, std::min(feltRect.width, feltRect.height));
    // Allow a wider outer expansion so we can reach the rail's outer edge.
    // (We still anchor via the seed band and bound via felt-proximity + bounding rect + dark mask.)
    const int outerBandPx = std::clamp(minDim / 12, 40, 240);
    const int seedBandPx = std::clamp(outerBandPx / 5, 6, 18);

    // Bounding rect around the felt to prevent any accidental far-field inclusion.
    // (This is extra defense on top of the distance-to-felt constraint.)
    const int pad = outerBandPx + 12;
    cv::Rect boundedRect(
        std::max(0, feltRect.x - pad),
        std::max(0, feltRect.y - pad),
        std::min(src.cols - std::max(0, feltRect.x - pad), feltRect.width + 2 * pad),
        std::min(src.rows - std::max(0, feltRect.y - pad), feltRect.height + 2 * pad)
    );
    boundedRect = boundedRect & cv::Rect(0, 0, src.cols, src.rows);
    cv::Mat boundedMask = cv::Mat::zeros(src.size(), CV_8UC1);
    boundedMask(boundedRect).setTo(255);

    // Non-felt area (255 outside expanded felt, 0 inside expanded felt)
    cv::Mat nonFeltMask;
    cv::bitwise_not(feltMaskExpanded, nonFeltMask);

    // Distance (in pixels) from each non-felt pixel to the nearest felt pixel (0 inside felt).
    cv::Mat distToFelt;
    cv::distanceTransform(nonFeltMask, distToFelt, cv::DIST_L2, 3);

    // bandMask(distMax) = (0 < dist <= distMax)
    auto makeBandMask = [&](int distMaxPx) -> cv::Mat {
        cv::Mat gt0, leBand, band;
        cv::compare(distToFelt, 0.0, gt0, cv::CMP_GT);
        cv::compare(distToFelt, (double)distMaxPx, leBand, cv::CMP_LE);
        cv::bitwise_and(gt0, leBand, band);
        return band;
    };

    // limitMask: where rails are allowed to exist (outside felt, within outer band, within bounding rect, and dark)
    cv::Mat limitBand = makeBandMask(outerBandPx);
    cv::Mat limitMask;
    cv::bitwise_and(limitBand, boundedMask, limitMask);
    cv::bitwise_and(limitMask, darkMask, limitMask);

    // seedMask: a very thin ring right next to felt (outside only). This anchors connectivity to the felt boundary.
    cv::Mat seedBand = makeBandMask(seedBandPx);
    cv::Mat seedMask;
    cv::bitwise_and(seedBand, boundedMask, seedMask);
    cv::bitwise_and(seedMask, darkMask, seedMask);

    // Step 5: Keep only connected components of limitMask that touch seedMask.
    // This expands outward to the rail's outer edge (within outerBandPx) without bleeding into unrelated dark regions.
    cv::Mat labels;
    const int num = cv::connectedComponents(limitMask, labels, 8, CV_32S);
    if (num <= 1) {
        return cv::Mat::zeros(src.size(), CV_8UC1);
    }

    std::vector<uint8_t> keep((size_t)num, 0);
    for (int y = 0; y < seedMask.rows; ++y) {
        const uchar* s = seedMask.ptr<uchar>(y);
        const int* lab = labels.ptr<int>(y);
        for (int x = 0; x < seedMask.cols; ++x) {
            if (s[x]) {
                const int id = lab[x];
                if (id >= 0 && id < num) keep[(size_t)id] = 1;
            }
        }
    }

    cv::Mat railMask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int y = 0; y < railMask.rows; ++y) {
        uchar* out = railMask.ptr<uchar>(y);
        const int* lab = labels.ptr<int>(y);
        for (int x = 0; x < railMask.cols; ++x) {
            const int id = lab[x];
            if (id > 0 && id < num && keep[(size_t)id]) {
                out[x] = 255;
            }
        }
    }

    // Step 6: Final cleanup (fill small gaps, remove speckles).
    // Keep kernels modest so we don't accidentally bridge across corners into non-rail dark regions.
    cv::Mat cleanupKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(railMask, railMask, cv::MORPH_CLOSE, cleanupKernel);
    cv::morphologyEx(railMask, railMask, cv::MORPH_OPEN, cleanupKernel);

    // Step 7: Drop tiny components if any remain (belt-and-suspenders).
    const int minKeepArea = std::max(400, (feltRect.area() / 900));
    cv::Mat stats, centroids;
    cv::Mat ccLabels;
    const int num2 = cv::connectedComponentsWithStats(railMask, ccLabels, stats, centroids, 8, CV_32S);
    if (num2 <= 1) {
        return railMask;
    }
    cv::Mat filtered = cv::Mat::zeros(railMask.size(), CV_8UC1);
    for (int i = 1; i < num2; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= minKeepArea) {
            filtered.setTo(255, ccLabels == i);
        }
    }
    cv::morphologyEx(filtered, filtered, cv::MORPH_CLOSE, cleanupKernel);

    // Step 8: Inward fill to guarantee the rail mask always meets the felt boundary.
    //
    // Motivation:
    // - Color/threshold noise can create a small "gap" between the detected rail pixels and the felt edge.
    // - Visually, it's better (and more correct) if the rail region always reaches the rail/felt interface.
    //
    // Strategy:
    // - Force-include a thin band immediately outside the (expanded) felt boundary.
    // - Because this band is defined by distance-to-felt (and explicitly excludes felt), it cannot invade the felt.
    {
        cv::Mat innerBand = makeBandMask(seedBandPx);
        cv::bitwise_and(innerBand, boundedMask, innerBand);
        cv::bitwise_or(filtered, innerBand, filtered);
        // Light close to stitch any 1-2px discontinuities along the boundary.
        cv::morphologyEx(filtered, filtered, cv::MORPH_CLOSE, cleanupKernel);
    }

    // Final hard guarantee: never include felt (even if cleanup bridged across the boundary).
    filtered.setTo(0, feltMaskExpanded);
    
    // Step 9: Fill holes (e.g., white diamond dots) so the rail mask is continuous (donut shape).
    // The diamonds are physically part of the rail surface, so they should be included in the mask.
    // We find internal contours (holes) and fill them.
    {
        cv::Mat maskWithHoles = filtered.clone();
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        // RETR_CCOMP retrieves all contours and organizes them into a two-level hierarchy
        cv::findContours(maskWithHoles, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty() && !hierarchy.empty()) {
            // Fill all holes (internal contours). In RETR_CCOMP hierarchy:
            // hierarchy[i][2] != -1 means contour i has children (holes inside it)
            // hierarchy[i][3] != -1 means contour i is a hole (child of another contour)
            for (size_t i = 0; i < contours.size(); ++i) {
                // If this contour is a hole (has a parent), fill it
                if (hierarchy[i][3] != -1) {
                    cv::drawContours(filtered, contours, static_cast<int>(i), cv::Scalar(255), -1);
                }
            }
        }
        
        // Ensure we still exclude felt after filling holes
        filtered.setTo(0, feltMaskExpanded);
    }
    
    // Step 10: Smooth edges by simplifying the contour with polygon approximation
    // This removes jagged edges while maintaining the general direction/shape of each border segment
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(filtered, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            // Find the largest contour (should be the rail donut shape)
            size_t largestIdx = 0;
            double largestArea = 0.0;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestIdx = i;
                }
            }
            
            // Simplify the contour using Douglas-Peucker algorithm
            // Epsilon controls the smoothing: larger = more smoothing (fewer points, straighter lines)
            // Using 2-3 pixels works well to smooth jagged edges while preserving shape
            const double epsilon = 2.5;  // pixels - tune this for more/less smoothing
            std::vector<cv::Point> smoothedContour;
            cv::approxPolyDP(contours[largestIdx], smoothedContour, epsilon, false);
            
            // Recreate the mask from the smoothed contour
            filtered = cv::Mat::zeros(filtered.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> fillContours;
            fillContours.push_back(smoothedContour);
            cv::fillPoly(filtered, fillContours, cv::Scalar(255));
            
            // Final guarantee: exclude felt area
            filtered.setTo(0, feltMaskExpanded);
        }
    }
    
    return filtered;
}

std::vector<cv::Point> detectRailOuterBoundary(const cv::Mat& railMask) {
    return computeOuterBoundaryQuadFromMask(railMask);
}

// Detect rail areas and return contours for visualization
std::vector<std::vector<cv::Point>> detectRailContours(const cv::Mat& src, const std::vector<cv::Point>& feltContour) {
    return detectRailContours(src, feltContour, RailParams{});
}

std::vector<std::vector<cv::Point>> detectRailContours(const cv::Mat& src, const std::vector<cv::Point>& feltContour, const RailParams& params) {
    cv::Mat railMask = detectRailMask(src, feltContour, params);
    
    // Find contours of rail areas
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(railMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter out very small contours (noise)
    std::vector<std::vector<cv::Point>> filteredContours;
    if (!feltContour.empty()) {
        cv::Rect feltRect = cv::boundingRect(feltContour);
        // Reduced minimum area threshold from *5 to *2 to catch smaller rail segments
        double minRailArea = (feltRect.width + feltRect.height) * 2;
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > minRailArea) {
                filteredContours.push_back(contour);
            }
        }
    }
    
    return filteredContours;
}

