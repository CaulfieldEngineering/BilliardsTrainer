#include "diamond_detection.h"
#include <algorithm>
#include <numeric>
#include <iostream>

// D) EMA threshold smoothing per rail (reduces frame-to-frame churn)
static float T_ema[4] = {0.0f, 0.0f, 0.0f, 0.0f};

// E) Temporal smoothing of output centers per (railIndex, positionOnRail)
struct TrackPt { cv::Point2f p; bool init = false; };
static TrackPt track[4][6];  // max 6 diamonds per rail

// Helper: Compute median of a vector
static double computeMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

// Helper function: Detect diamonds on a single rail mask using improved algorithm
static std::vector<Diamond> detectDiamondsOnRail(
    const cv::Mat& rectifiedGray,
    const cv::Mat& railMask,
    const DiamondParams& params,
    int railIndex,
    int expectedCount,
    cv::Mat* debugTophat = nullptr,
    cv::Mat* debugBin = nullptr
) {
    std::vector<Diamond> diamonds;

    if (rectifiedGray.empty() || railMask.empty()) {
        std::cout << "[Diamond Rail " << railIndex << "] Input empty\n";
        return diamonds;
    }

    // A) Extract rail ROI with eroded mask (avoid boundary edges)
    cv::Rect roi = cv::boundingRect(railMask);

    // Validate ROI
    if (roi.width <= 0 || roi.height <= 0 ||
        roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > rectifiedGray.cols ||
        roi.y + roi.height > rectifiedGray.rows) {
        std::cout << "[Diamond Rail " << railIndex << "] Invalid ROI\n";
        return diamonds;
    }

    cv::Mat roiGray = rectifiedGray(roi);
    cv::Mat roiMask = railMask(roi).clone();

    int nzMask = cv::countNonZero(roiMask);
    std::cout << "[Diamond Rail " << railIndex << "] ROI=" << roi << " nzMask=" << nzMask << "\n";

    // Use distance transform to create inner mask (avoids killing mask entirely)
    cv::Mat roiMaskInner;
    cv::Mat dist;
    cv::distanceTransform(roiMask, dist, cv::DIST_L2, 3);
    roiMaskInner = dist > 2.0f;  // 2 px away from edge
    roiMaskInner.convertTo(roiMaskInner, CV_8U, 255);

    int nzInner = cv::countNonZero(roiMaskInner);
    std::cout << "[Diamond Rail " << railIndex << "] nzInner=" << nzInner << " (distance transform)\n";

    if (nzInner == 0) {
        std::cout << "[Diamond Rail " << railIndex << "] Mask too small after distance transform\n";
        return diamonds;
    }

    // B) Enhance diamonds (small bright blobs) using morphological top-hat
    // Kernel size should be similar to diamond diameter in rectified px (try 11x11 or 15x15)
    cv::Mat tophat;
    cv::Mat tophatKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
    cv::morphologyEx(roiGray, tophat, cv::MORPH_TOPHAT, tophatKernel);

    // Get tophat stats
    double minVal, maxVal;
    cv::minMaxLoc(tophat, &minVal, &maxVal);
    cv::Scalar meanVal = cv::mean(tophat, roiMaskInner);
    std::cout << "[Diamond Rail " << railIndex << "] tophat min/max/mean="
              << minVal << "/" << maxVal << "/" << meanVal[0] << "\n";

    // Save debug tophat if requested
    if (debugTophat) {
        tophat.copyTo(*debugTophat);
    }

    // C) Threshold ONLY inside roiMaskInner
    // Build a masked image for thresholding: tophat with pixels outside roiMaskInner set to 0
    cv::Mat masked = cv::Mat::zeros(tophat.size(), tophat.type());
    tophat.copyTo(masked, roiMaskInner);

    int nzMasked = cv::countNonZero(masked);
    std::cout << "[Diamond Rail " << railIndex << "] nzMasked=" << nzMasked << "\n";

    // Percentile-based threshold: more robust than "% of max"
    cv::Mat bin;

    // Collect masked tophat values
    std::vector<uchar> vals;
    vals.reserve(nzInner);
    for (int y = 0; y < masked.rows; y++) {
        for (int x = 0; x < masked.cols; x++) {
            if (roiMaskInner.at<uchar>(y, x)) {
                vals.push_back(masked.at<uchar>(y, x));
            }
        }
    }

    if (vals.empty()) {
        std::cout << "[Diamond Rail " << railIndex << "] No masked pixels\n";
        return diamonds;
    }

    // Use 90th percentile as threshold
    size_t p90_idx = (size_t)(0.90 * vals.size());
    if (p90_idx >= vals.size()) p90_idx = vals.size() - 1;

    std::nth_element(vals.begin(), vals.begin() + p90_idx, vals.end());
    int T = std::max(5, (int)vals[p90_idx]);

    // D) Apply EMA smoothing to threshold (reduces frame-to-frame churn)
    float alpha = 0.2f; // smaller = steadier
    if (T_ema[railIndex] <= 0) T_ema[railIndex] = (float)T;
    T_ema[railIndex] = (1.0f - alpha) * T_ema[railIndex] + alpha * (float)T;
    int Tstable = (int)std::round(T_ema[railIndex]);

    cv::threshold(masked, bin, Tstable, 255, cv::THRESH_BINARY);

    int nzBinBeforeMorph = cv::countNonZero(bin);
    std::cout << "[Diamond Rail " << railIndex << "] p90thresh=" << T
              << " ema=" << Tstable
              << " nzBin (before morph)=" << nzBinBeforeMorph << "\n";

    // D) Clean with light morphology - TEMPORARILY DISABLED until stable
    // cv::Mat cleanKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    // cv::morphologyEx(bin, bin, cv::MORPH_OPEN, cleanKernel);
    // cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, cleanKernel);

    int nzBin = cv::countNonZero(bin);
    std::cout << "[Diamond Rail " << railIndex << "] nzBin (after morph)=" << nzBin << "\n";

    // Save debug bin if requested
    if (debugBin) {
        bin.copyTo(*debugBin);
    }

    if (nzBin == 0) {
        std::cout << "[Diamond Rail " << railIndex << "] bin is empty, no features detected\n";
        return diamonds;
    }

    // E) Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "[Diamond Rail " << railIndex << "] Found " << contours.size() << " contours\n";

    // F) Extract all blobs with geometry gates
    std::vector<Diamond> candidates;

    // Helper: Distance from mask edge at a point
    auto distAt = [&](const cv::Point2f& c) -> float {
        int x = std::clamp((int)std::round(c.x) - roi.x, 0, dist.cols - 1);
        int y = std::clamp((int)std::round(c.y) - roi.y, 0, dist.rows - 1);
        return dist.at<float>(y, x);
    };

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        cv::Rect bbox = cv::boundingRect(contour);

        // A1) Geometry gates: bbox shape filters
        if (bbox.width < 3 || bbox.height < 3) continue;          // tiny noise
        if (bbox.width > 40 || bbox.height > 40) continue;        // stops huge boxes

        double boxArea = (double)bbox.width * bbox.height;
        double fillRatio = area / std::max(1.0, boxArea);
        double aspect = (double)bbox.width / std::max(1, bbox.height);

        if (aspect < 0.4 || aspect > 2.5) continue;               // not extremely elongated
        if (fillRatio < 0.25) continue;                           // rails/seams often low fill

        // Compute center
        cv::Moments m = cv::moments(contour);
        if (m.m00 == 0) continue;

        cv::Point2f centerLocal(m.m10 / m.m00, m.m01 / m.m00);
        cv::Point2f centerGlobal(roi.x + centerLocal.x, roi.y + centerLocal.y);

        // A2) Reject candidates too close to rail mask edge
        if (distAt(centerGlobal) < 4.0f) continue;   // 4px away from mask edge

        // A3) Do not allow candidates in pocket zones (end margins)
        int padEnd = 25; // px margin at rail ends
        if (railIndex == 0 || railIndex == 2) { // top/bottom: use x
            float x = centerLocal.x;
            if (x < padEnd || x > (roi.width - padEnd)) continue;
        } else { // left/right: use y
            float y = centerLocal.y;
            if (y < padEnd || y > (roi.height - padEnd)) continue;
        }

        Diamond d;
        d.center = cv::Point2f(
            roi.x + m.m10 / m.m00,
            roi.y + m.m01 / m.m00
        );

        // Convert contour back to full-image coordinates
        std::vector<cv::Point> fullContour;
        for (const auto& pt : contour) {
            fullContour.push_back(cv::Point(pt.x + roi.x, pt.y + roi.y));
        }
        d.contour = fullContour;

        d.boundingBox = cv::Rect(
            bbox.x + roi.x,
            bbox.y + roi.y,
            bbox.width,
            bbox.height
        );
        d.railIndex = railIndex;
        d.radius = std::sqrt(area / CV_PI);
        d.positionOnRail = -1;  // Will be assigned later

        candidates.push_back(d);
    }

    std::cout << "[Diamond Rail " << railIndex << "] Candidates before pruning: " << candidates.size() << "\n";

    if (candidates.empty()) {
        return diamonds;
    }

    // G) RANKED SELECTION: Use tophat strength + spacing constraints (robust to junk blobs)

    // B) Helper: 7x7 windowed sum score (far less jittery than single pixel)
    auto scoreAt = [&](const cv::Point2f& c) -> int {
        int cx = std::clamp((int)std::round(c.x) - roi.x, 0, tophat.cols - 1);
        int cy = std::clamp((int)std::round(c.y) - roi.y, 0, tophat.rows - 1);
        int r = 3; // 7x7 window

        int sum = 0;
        for (int dy = -r; dy <= r; dy++) {
            int y = std::clamp(cy + dy, 0, tophat.rows - 1);
            for (int dx = -r; dx <= r; dx++) {
                int x = std::clamp(cx + dx, 0, tophat.cols - 1);
                if (roiMaskInner.at<uchar>(y, x)) {
                    sum += (int)tophat.at<uchar>(y, x);
                }
            }
        }
        return sum;
    };

    // Candidate structure for ranking
    struct Cand {
        Diamond d;
        int score;    // tophat intensity at center
        double area;
    };

    // Build ranked candidate list
    std::vector<Cand> ranked;
    ranked.reserve(candidates.size());
    for (auto& d : candidates) {
        double area = cv::contourArea(d.contour);
        int score = scoreAt(d.center);  // Use windowed sum instead of single pixel
        ranked.push_back({d, score, area});
    }

    // Optional: basic area clamp (no median - just min/max bounds)
    // Force permissive bounds for bring-up (10-2000 px²)
    const double minAreaPermissive = 10.0;
    const double maxAreaPermissive = 2000.0;
    ranked.erase(std::remove_if(ranked.begin(), ranked.end(), [&](const Cand& c) {
        return c.area < minAreaPermissive || c.area > maxAreaPermissive;
    }), ranked.end());

    std::cout << "[Diamond Rail " << railIndex << "] After area bounds: " << ranked.size() << "\n";

    // Sort by tophat score (descending), then by area (descending)
    std::sort(ranked.begin(), ranked.end(), [](const Cand& a, const Cand& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.area > b.area;
    });

    // Debug: print top 10 ranked candidates
    int numToPrint = std::min((int)ranked.size(), 10);
    for (int i = 0; i < numToPrint; i++) {
        std::cout << "  Ranked[" << i << "]: score=" << ranked[i].score
                  << " area=" << ranked[i].area
                  << " center=(" << ranked[i].d.center.x << "," << ranked[i].d.center.y << ")\n";
    }

    // C) Non-max suppression with rail-specific expected spacing
    float axisLen = (railIndex == 0 || railIndex == 2) ? (float)roi.width : (float)roi.height;
    float expectedSpacing = axisLen / (expectedCount + 1);  // rough expected spacing
    float minSpacingPx = 0.55f * expectedSpacing;           // tighter + consistent

    std::cout << "[Diamond Rail " << railIndex << "] axisLen=" << axisLen
              << " expectedSpacing=" << expectedSpacing
              << " minSpacingPx=" << minSpacingPx << "\n";

    std::vector<Diamond> selected;
    for (const auto& c : ranked) {
        bool tooClose = false;
        for (const auto& s : selected) {
            // Measure distance along rail axis
            float distAxis = (railIndex == 0 || railIndex == 2)
                ? std::abs(c.d.center.x - s.center.x)  // top/bottom: horizontal spacing
                : std::abs(c.d.center.y - s.center.y); // left/right: vertical spacing

            if (distAxis < minSpacingPx) {
                tooClose = true;
                break;
            }
        }

        if (!tooClose) {
            selected.push_back(c.d);
        }

        // Stop once we have enough
        if ((int)selected.size() >= expectedCount) {
            break;
        }
    }

    diamonds = selected;
    std::cout << "[Diamond Rail " << railIndex << "] Final selected: " << diamonds.size()
              << " (expected " << expectedCount << ")\n";

    return diamonds;
}

// Helper function: Sort diamonds along a rail and assign position indices
static void assignDiamondPositions(std::vector<Diamond>& diamonds, int railIndex) {
    if (diamonds.empty()) return;

    // Sort based on rail orientation
    if (railIndex == 0 || railIndex == 2) {
        // Top or bottom rail: sort left to right (by x)
        std::sort(diamonds.begin(), diamonds.end(),
                  [](const Diamond& a, const Diamond& b) { return a.center.x < b.center.x; });
    } else {
        // Left or right rail: sort top to bottom (by y)
        std::sort(diamonds.begin(), diamonds.end(),
                  [](const Diamond& a, const Diamond& b) { return a.center.y < b.center.y; });
    }

    // Assign position indices
    for (size_t i = 0; i < diamonds.size(); i++) {
        diamonds[i].positionOnRail = static_cast<int>(i);
    }
}

// Main diamond detection function
DiamondDetectionResult detectDiamonds(
    const cv::Mat& rectifiedBgr,
    const std::vector<cv::Mat>& railMasks,
    const DiamondParams& params
) {
    DiamondDetectionResult result;
    result.ok = false;
    result.numDiamondsDetected = 0;
    result.diamondsPerRail = {0, 0, 0, 0};

    std::cout << "\n=== Diamond Detection Start ===\n";

    // Validate inputs
    if (rectifiedBgr.empty() || railMasks.size() != 4) {
        std::cout << "[Diamond] Invalid inputs\n";
        return result;
    }

    // Convert to grayscale once
    cv::Mat rectifiedGray;
    cv::cvtColor(rectifiedBgr, rectifiedGray, cv::COLOR_BGR2GRAY);
    std::cout << "[Diamond] Grayscale conversion done, size=" << rectifiedGray.size() << "\n";

    // Expected diamond counts per rail
    // Rail order: [top=0, right=1, bottom=2, left=3]
    // Top/bottom (short) rails: 3 diamonds each
    // Left/right (long) rails: 6 diamonds each
    std::array<int, 4> expectedCounts = {
        params.shortRailDiamonds,  // top
        params.longRailDiamonds,   // right
        params.shortRailDiamonds,  // bottom
        params.longRailDiamonds    // left
    };

    // Detect diamonds on each rail
    for (int i = 0; i < 4; i++) {
        if (railMasks[i].empty()) {
            std::cout << "[Diamond] Rail " << i << " mask is empty\n";
            continue;
        }

        std::vector<Diamond> railDiamonds = detectDiamondsOnRail(
            rectifiedGray, railMasks[i], params, i, expectedCounts[i]
        );

        // Assign positions along the rail
        assignDiamondPositions(railDiamonds, i);

        // E) Apply temporal smoothing to reduce 1-3 px jitter
        float beta = 0.15f; // 0.1-0.2, smaller = steadier
        for (auto& d : railDiamonds) {
            int k = d.positionOnRail;
            if (k < 0 || k >= 6) continue;

            auto& t = track[i][k];
            if (!t.init) {
                t.p = d.center;
                t.init = true;
            }
            t.p = (1.0f - beta) * t.p + beta * d.center;
            d.center = t.p;

            // Recompute bounding box from smoothed center
            // (Keep original radius/contour, just update bbox center)
            int halfW = d.boundingBox.width / 2;
            int halfH = d.boundingBox.height / 2;
            d.boundingBox.x = (int)std::round(d.center.x) - halfW;
            d.boundingBox.y = (int)std::round(d.center.y) - halfH;
        }

        // Add to results
        result.diamonds.insert(result.diamonds.end(), railDiamonds.begin(), railDiamonds.end());
        result.diamondsPerRail[i] = static_cast<int>(railDiamonds.size());
    }

    result.numDiamondsDetected = static_cast<int>(result.diamonds.size());

    std::cout << "[Diamond] Total detected: " << result.numDiamondsDetected << "\n";
    std::cout << "[Diamond] Per rail: [" << result.diamondsPerRail[0] << ", "
              << result.diamondsPerRail[1] << ", "
              << result.diamondsPerRail[2] << ", "
              << result.diamondsPerRail[3] << "]\n";
    std::cout << "=== Diamond Detection End ===\n\n";

    // Mark ok if we have any diamonds
    result.ok = !result.diamonds.empty();

    return result;
}

// Helper function: Project a single point from rectified to perspective view
static cv::Point2f projectPoint(const cv::Point2f& pt, const cv::Mat& Hinv) {
    if (Hinv.empty()) return pt;

    // Convert to homogeneous coordinates
    cv::Mat ptMat = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);

    // Apply inverse homography
    cv::Mat projectedMat = Hinv * ptMat;

    // Convert back to 2D
    double w = projectedMat.at<double>(2, 0);
    if (std::abs(w) < 1e-6) return pt;  // Avoid division by zero

    float x = static_cast<float>(projectedMat.at<double>(0, 0) / w);
    float y = static_cast<float>(projectedMat.at<double>(1, 0) / w);

    return cv::Point2f(x, y);
}

// Project diamond detection results from rectified to perspective view
DiamondDetectionResult projectDiamondsToPerspective(
    const DiamondDetectionResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
) {
    DiamondDetectionResult result = rectifiedResult;

    if (Hinv.empty()) {
        return result;
    }

    // Project each diamond's center and contour
    for (auto& diamond : result.diamonds) {
        // Project center
        diamond.center = projectPoint(diamond.center, Hinv);

        // Project contour points
        std::vector<cv::Point> projectedContour;
        for (const auto& pt : diamond.contour) {
            cv::Point2f projected = projectPoint(cv::Point2f(pt.x, pt.y), Hinv);
            projectedContour.push_back(cv::Point(static_cast<int>(projected.x),
                                                  static_cast<int>(projected.y)));
        }
        diamond.contour = projectedContour;

        // Recompute bounding box in perspective space
        if (!projectedContour.empty()) {
            diamond.boundingBox = cv::boundingRect(projectedContour);
        }
    }

    return result;
}

// Draw diamond overlay on an image
void drawDiamondOverlay(cv::Mat& img, const DiamondDetectionResult& result, const DiamondParams& params) {
    if (img.empty() || result.diamonds.empty()) {
        return;
    }

    // TEMPORARY: Use high-contrast yellow for debugging visibility
    const cv::Scalar debugColor = cv::Scalar(0, 255, 255);  // BGR: Yellow
    const int debugThickness = 3;

    for (const auto& diamond : result.diamonds) {
        // Draw bounding box for debugging
        cv::rectangle(img, diamond.boundingBox, debugColor, debugThickness);

        // Draw circle outline (always visible)
        cv::circle(img, diamond.center, static_cast<int>(diamond.radius),
                  debugColor, debugThickness);

        // Draw center cross
        int crossSize = 5;
        cv::line(img, cv::Point(diamond.center.x - crossSize, diamond.center.y),
                cv::Point(diamond.center.x + crossSize, diamond.center.y), debugColor, 2);
        cv::line(img, cv::Point(diamond.center.x, diamond.center.y - crossSize),
                cv::Point(diamond.center.x, diamond.center.y + crossSize), debugColor, 2);
    }
}

// Draw diamond centers on an image
void drawDiamondCenters(cv::Mat& img, const DiamondDetectionResult& result, const DiamondParams& params) {
    if (img.empty() || result.diamonds.empty()) {
        return;
    }

    for (const auto& diamond : result.diamonds) {
        // Draw a small cross at the center
        int size = 3;
        cv::line(img, cv::Point(diamond.center.x - size, diamond.center.y),
                cv::Point(diamond.center.x + size, diamond.center.y), params.color, 1);
        cv::line(img, cv::Point(diamond.center.x, diamond.center.y - size),
                cv::Point(diamond.center.x, diamond.center.y + size), params.color, 1);

        // Optionally draw position label
        // std::string label = std::to_string(diamond.positionOnRail);
        // cv::putText(img, label, diamond.center, cv::FONT_HERSHEY_SIMPLEX, 0.4, params.color, 1);
    }
}
