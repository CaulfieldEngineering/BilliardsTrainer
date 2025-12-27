#include "diamond_detection.h"
#include "../FeltDetection/felt_detection.h"
#include "../RailDetection/rail_detection.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <array>
#include <limits>

// Owning definition for debug outputs (declared extern in the header)
std::vector<std::pair<std::string, cv::Mat>> g_lastDiamondDebugImages;

// -------------------------------------------------------------------------------------------------
// Temporary debug geometry helpers
// -------------------------------------------------------------------------------------------------

// Select points near the "extreme" along one axis.
// - axis = 0 -> x, axis = 1 -> y
// - selectLow = true selects points near min(axis); false selects near max(axis)
// - fraction controls how thick the band is (e.g. 0.25 = outer 25% of the span)
static std::vector<cv::Point2f> selectExtremeBand(
    const std::vector<cv::Point2f>& pts,
    int axis,
    bool selectLow,
    float fraction)
{
    std::vector<cv::Point2f> out;
    if (pts.empty()) return out;

    fraction = std::clamp(fraction, 0.01f, 0.49f);

    float minV = std::numeric_limits<float>::infinity();
    float maxV = -std::numeric_limits<float>::infinity();
    for (const auto& p : pts) {
        const float v = (axis == 0) ? p.x : p.y;
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }

    const float span = std::max(1.0f, maxV - minV);
    const float band = span * fraction;

    const float lo = minV + band;
    const float hi = maxV - band;

    out.reserve(pts.size());
    for (const auto& p : pts) {
        const float v = (axis == 0) ? p.x : p.y;
        if (selectLow) {
            if (v <= lo) out.push_back(p);
        } else {
            if (v >= hi) out.push_back(p);
        }
    }
    return out;
}

// Draw an "infinite" line (clipped to the image rectangle) from a set of points.
// Uses cv::fitLine to obtain a best-fit line in least-squares sense.
static void drawInfiniteBestFitLine(
    cv::Mat& img,
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    int thickness)
{
    if (img.empty()) return;
    if ((int)pts.size() < 2) return;

    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);

    const float vx = line[0];
    const float vy = line[1];
    const float x0 = line[2];
    const float y0 = line[3];

    const int w = img.cols;
    const int h = img.rows;

    // Find intersections between the infinite line and the image rectangle.
    // Parametric form: (x, y) = (x0, y0) + t * (vx, vy)
    std::vector<cv::Point2f> hits;
    hits.reserve(4);

    auto addIfInside = [&](float x, float y) {
        if (x >= 0.0f && x <= (float)(w - 1) && y >= 0.0f && y <= (float)(h - 1)) {
            hits.emplace_back(x, y);
        }
    };

    // Intersect with x = 0 and x = w-1
    if (std::abs(vx) > 1e-6f) {
        float tL = (0.0f - x0) / vx;
        addIfInside(0.0f, y0 + tL * vy);
        float tR = ((float)(w - 1) - x0) / vx;
        addIfInside((float)(w - 1), y0 + tR * vy);
    }

    // Intersect with y = 0 and y = h-1
    if (std::abs(vy) > 1e-6f) {
        float tT = (0.0f - y0) / vy;
        addIfInside(x0 + tT * vx, 0.0f);
        float tB = ((float)(h - 1) - y0) / vy;
        addIfInside(x0 + tB * vx, (float)(h - 1));
    }

    if (hits.size() < 2) return;

    // Choose the farthest pair of intersection points (stable even if we collected 3-4 points).
    float bestD2 = -1.0f;
    cv::Point2f a = hits[0], b = hits[1];
    for (size_t i = 0; i < hits.size(); i++) {
        for (size_t j = i + 1; j < hits.size(); j++) {
            const float dx = hits[i].x - hits[j].x;
            const float dy = hits[i].y - hits[j].y;
            const float d2 = dx * dx + dy * dy;
            if (d2 > bestD2) {
                bestD2 = d2;
                a = hits[i];
                b = hits[j];
            }
        }
    }

    cv::line(
        img,
        cv::Point((int)std::lround(a.x), (int)std::lround(a.y)),
        cv::Point((int)std::lround(b.x), (int)std::lround(b.y)),
        color,
        std::max(1, thickness),
        cv::LINE_AA);
}

// Forward declarations for helpers used by later debug/validation utilities.
static float distancePointToInfiniteLine(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);

// Draw an infinite line (clipped to the image rectangle) that passes through two anchor points.
// This guarantees the drawn line intersects those two diamond centers.
static void drawInfiniteLineThroughTwoPoints(
    cv::Mat& img,
    const cv::Point2f& p0,
    const cv::Point2f& p1,
    const cv::Scalar& color,
    int thickness)
{
    if (img.empty()) return;

    const cv::Point2f v = p1 - p0;
    const float vLen = std::sqrt(v.dot(v));
    if (vLen <= 1e-6f) return;

    const float vx = v.x / vLen;
    const float vy = v.y / vLen;

    const int w = img.cols;
    const int h = img.rows;

    // Parametric form: (x, y) = p0 + t * (vx, vy)
    std::vector<cv::Point2f> hits;
    hits.reserve(4);

    auto addIfInside = [&](float x, float y) {
        if (x >= 0.0f && x <= (float)(w - 1) && y >= 0.0f && y <= (float)(h - 1)) {
            hits.emplace_back(x, y);
        }
    };

    if (std::abs(vx) > 1e-6f) {
        float tL = (0.0f - p0.x) / vx;
        addIfInside(0.0f, p0.y + tL * vy);
        float tR = ((float)(w - 1) - p0.x) / vx;
        addIfInside((float)(w - 1), p0.y + tR * vy);
    }

    if (std::abs(vy) > 1e-6f) {
        float tT = (0.0f - p0.y) / vy;
        addIfInside(p0.x + tT * vx, 0.0f);
        float tB = ((float)(h - 1) - p0.y) / vy;
        addIfInside(p0.x + tB * vx, (float)(h - 1));
    }

    if (hits.size() < 2) return;

    // Choose farthest pair for stability.
    float bestD2 = -1.0f;
    cv::Point2f a = hits[0], b = hits[1];
    for (size_t i = 0; i < hits.size(); i++) {
        for (size_t j = i + 1; j < hits.size(); j++) {
            const float dx = hits[i].x - hits[j].x;
            const float dy = hits[i].y - hits[j].y;
            const float d2 = dx * dx + dy * dy;
            if (d2 > bestD2) {
                bestD2 = d2;
                a = hits[i];
                b = hits[j];
            }
        }
    }

    cv::line(
        img,
        cv::Point((int)std::lround(a.x), (int)std::lround(a.y)),
        cv::Point((int)std::lround(b.x), (int)std::lround(b.y)),
        color,
        std::max(1, thickness),
        cv::LINE_AA);
}

// Choose an anchored rail line from a set of points:
// - pick a pair of points (anchors) such that the line passes through BOTH
// - maximize "inliers" (other points close to the line), so a third point can validate the fit
// Returns true if it found a line through at least 2 points.
static bool drawAnchoredRailLineFromPoints(
    cv::Mat& img,
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    int thickness)
{
    if ((int)pts.size() < 2) return false;

    // For a small number of rail points (<=6 typically), brute force all pairs.
    const float inlierThreshPx = 16.0f;

    int bestI = 0;
    int bestJ = 1;
    int bestInliers = 2;
    float bestResidual = std::numeric_limits<float>::infinity();

    for (int i = 0; i < (int)pts.size(); i++) {
        for (int j = i + 1; j < (int)pts.size(); j++) {
            const cv::Point2f a = pts[i];
            const cv::Point2f b = pts[j];
            if (std::sqrt((b - a).dot(b - a)) < 20.0f) continue; // avoid near-duplicate anchors

            int inliers = 0;
            float residual = 0.0f;
            for (int k = 0; k < (int)pts.size(); k++) {
                const float d = distancePointToInfiniteLine(pts[k], a, b);
                if (d <= inlierThreshPx) {
                    inliers++;
                    residual += d;
                }
            }

            // Prefer: more inliers; then lower residual.
            if (inliers > bestInliers || (inliers == bestInliers && residual < bestResidual)) {
                bestInliers = inliers;
                bestResidual = residual;
                bestI = i;
                bestJ = j;
            }
        }
    }

    drawInfiniteLineThroughTwoPoints(img, pts[bestI], pts[bestJ], color, thickness);
    return true;
}

static float distancePointToSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f ab = b - a;
    const float ab2 = ab.dot(ab);
    if (ab2 <= 1e-6f) {
        const cv::Point2f d = p - a;
        return std::sqrt(d.dot(d));
    }

    const float t = std::clamp((p - a).dot(ab) / ab2, 0.0f, 1.0f);
    const cv::Point2f proj = a + t * ab;
    const cv::Point2f d = p - proj;
    return std::sqrt(d.dot(d));
}

// Distance from a point to an infinite line defined by segment endpoints (a->b).
// This avoids the "corner stealing" effect that happens with point-to-segment distance when
// diamonds are near a rail corner: segment distance can prefer the adjacent edge endpoint.
static float distancePointToInfiniteLine(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f ab = b - a;
    const float abLen = std::sqrt(ab.dot(ab));
    if (abLen <= 1e-6f) {
        const cv::Point2f d = p - a;
        return std::sqrt(d.dot(d));
    }
    // Perpendicular distance = |(b-a) x (p-a)| / |b-a|
    const cv::Point2f ap = p - a;
    const float cross = std::abs(ab.x * ap.y - ab.y * ap.x);
    return cross / abLen;
}

static float projectPointToLineScalar(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    // Project onto the line direction (a->b). Returned scalar is in pixels along the line.
    const cv::Point2f ab = b - a;
    const float abLen = std::sqrt(ab.dot(ab));
    if (abLen <= 1e-6f) return 0.0f;
    const cv::Point2f dir = ab * (1.0f / abLen);
    return (p - a).dot(dir);
}

// Choose the best subset of points for a rail.
// - Uses a combined score:
//   - closeness to the rail line (perpendicular distance)
//   - spacing consistency along the rail
//     - short rail: 3 points roughly equally spaced
//     - long rail: 6 points with one larger "pocket gap" between the 3rd and 4th point
struct RailSubsetResult {
    std::vector<int> indices;  // indices into the original `diamonds` array
    float score = std::numeric_limits<float>::infinity();
    int requiredCount = 0;
    int candidateCount = 0;
};

static RailSubsetResult chooseBestRailSubset(
    const std::vector<int>& assignedIndices,
    const std::vector<cv::Point2f>& allPoints,
    const cv::Point2f& lineA,
    const cv::Point2f& lineB,
    int requiredCount)
{
    RailSubsetResult out;
    out.requiredCount = requiredCount;
    out.candidateCount = (int)assignedIndices.size();
    if (requiredCount <= 0) {
        out.indices.clear();
        out.score = 1e9f;
        return out;
    }
    if ((int)assignedIndices.size() < requiredCount) {
        // Not enough candidates. Return what we have, but penalize heavily so hypothesis selection
        // prefers assignments that can meet the expected counts.
        out.indices = assignedIndices;
        const int deficit = requiredCount - (int)assignedIndices.size();
        out.score = 1e7f + (float)deficit * 1e6f;
        return out;
    }
    if ((int)assignedIndices.size() == requiredCount) {
        out.indices = assignedIndices;
        out.score = 0.0f; // perfect count; scoring will be handled by hypothesis comparison anyway
        return out;
    }

    struct Candidate {
        int idx = -1;
        float dist = 0.0f;   // perpendicular distance to line
        float t = 0.0f;      // projection scalar along line
    };

    std::vector<Candidate> cands;
    cands.reserve(assignedIndices.size());
    for (int idx : assignedIndices) {
        const cv::Point2f p = allPoints[idx];
        cands.push_back(Candidate{
            idx,
            distancePointToInfiniteLine(p, lineA, lineB),
            projectPointToLineScalar(p, lineA, lineB)
        });
    }

    // Sort by closeness to line and only consider the top M for combinatorial selection.
    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.dist < b.dist;
    });

    const int M = std::min<int>((int)cands.size(), 12); // 12 keeps brute force cheap (C(12,6)=924)
    cands.resize(M);

    // Brute force combinations (small M).
    std::vector<int> bestCandIdxs;
    float bestScore = std::numeric_limits<float>::infinity();

    std::vector<int> pick;
    pick.reserve(requiredCount);

    auto scoreCombo = [&](const std::vector<int>& comboCandIdxs) -> float {
        // dist score: prefer points very close to the rail line
        float distScore = 0.0f;
        std::vector<float> ts;
        ts.reserve(comboCandIdxs.size());
        for (int ci : comboCandIdxs) {
            distScore += cands[ci].dist;
            ts.push_back(cands[ci].t);
        }

        std::sort(ts.begin(), ts.end());
        if (ts.size() < 2) return distScore;

        std::vector<float> diffs;
        diffs.reserve(ts.size() - 1);
        for (size_t i = 1; i < ts.size(); i++) {
            diffs.push_back(std::abs(ts[i] - ts[i - 1]));
        }

        // Reject near-duplicates (very small spacing along the rail).
        for (float d : diffs) {
            if (d < 20.0f) {
                return distScore + 1e6f; // huge penalty
            }
        }

        auto normalizedVariance = [&](const std::vector<float>& xs) -> float {
            if (xs.empty()) return 0.0f;
            float mean = 0.0f;
            for (float x : xs) mean += x;
            mean /= (float)xs.size();
            mean = std::max(1.0f, mean);
            float var = 0.0f;
            for (float x : xs) {
                const float z = (x / mean) - 1.0f;
                var += z * z;
            }
            var /= (float)xs.size();
            return var;
        };

        float spacingScore = 0.0f;
        if ((int)comboCandIdxs.size() == 3) {
            // Want 2 gaps about equal
            spacingScore = normalizedVariance(diffs);
        } else if ((int)comboCandIdxs.size() == 6) {
            // Want 5 gaps: 4 roughly equal + 1 larger pocket gap, ideally between 3rd and 4th.
            int maxIdx = 0;
            float maxGap = diffs[0];
            for (int i = 1; i < (int)diffs.size(); i++) {
                if (diffs[i] > maxGap) { maxGap = diffs[i]; maxIdx = i; }
            }

            std::vector<float> smallDiffs;
            smallDiffs.reserve(diffs.size() - 1);
            for (int i = 0; i < (int)diffs.size(); i++) {
                if (i == maxIdx) continue;
                smallDiffs.push_back(diffs[i]);
            }

            // Pocket gap should be notably larger than the typical spacing.
            float meanSmall = 0.0f;
            for (float d : smallDiffs) meanSmall += d;
            meanSmall /= (float)smallDiffs.size();
            meanSmall = std::max(1.0f, meanSmall);

            const float ratio = maxGap / meanSmall;
            float ratioPenalty = 0.0f;
            if (ratio < 1.6f) {
                ratioPenalty = (1.6f - ratio) * 2.0f; // normalized-ish
            }

            // The pocket is between the 3rd and 4th diamond => gap index ~2 in a 6-point list.
            const float pocketPosPenalty = std::abs(maxIdx - 2) * 0.15f;

            spacingScore =
                normalizedVariance(smallDiffs) +
                ratioPenalty +
                pocketPosPenalty;
        }

        // Weight spacing much higher than dist once we're in the "close enough" set.
        // distScore is in pixels; spacingScore is dimensionless.
        return distScore + (spacingScore * 40.0f);
    };

    // Combination recursion over [0..M)
    std::function<void(int, int)> rec = [&](int start, int left) {
        if (left == 0) {
            const float s = scoreCombo(pick);
            if (s < bestScore) {
                bestScore = s;
                bestCandIdxs = pick;
            }
            return;
        }
        for (int i = start; i <= M - left; i++) {
            pick.push_back(i);
            rec(i + 1, left - 1);
            pick.pop_back();
        }
    };

    rec(0, requiredCount);

    out.indices.clear();
    out.indices.reserve(bestCandIdxs.size());
    for (int ci : bestCandIdxs) out.indices.push_back(cands[ci].idx);
    out.score = bestScore;
    return out;
}

void detectDiamonds(
    const cv::Mat& src,
    cv::Mat& dst,
    bool showDiamonds,
    const DiamondDetectionParams& diamondParams,
    const FeltParams& feltParams,
    const RailParams& railParams,
    cv::Mat* outProcessingImage)
{
    // Clear debug outputs on every call so the UI does not reuse stale images.
    g_lastDiamondDebugImages.clear();

    if (src.empty()) {
        if (outProcessingImage) outProcessingImage->release();
        return;
    }

    // Initialize the destination buffer if the caller passed an empty Mat.
    if (dst.empty()) {
        dst = src.clone();
    }

    if (!showDiamonds) {
        if (outProcessingImage) outProcessingImage->release();
        return;
    }

    // ============================================================================
    // Signal Processing Pipeline - Order of Operations:
    // ============================================================================
    // Step 1: Get felt contour and rail mask (spatial filtering - isolate rail region)
    // Step 2: Isolate rail region using rail mask
    // Step 3: Convert to grayscale (color space conversion)
    // Step 4: Morphological enhancement (top-hat/black-hat) - enhances small features
    // Step 5: Threshold (binary segmentation) - uses threshold parameter value
    // Step 6: Morphological cleanup (opening - removes noise) - small 3x3 kernel
    // Step 7: Find contours (boundary extraction)
    // Step 8: Filter contours by area and circularity (feature validation)
    // Step 9: Draw detected diamonds (visualization)
    // ============================================================================

    // Step 1: Get felt contour and rail mask
    std::vector<cv::Point> feltContour = detectFeltContour(src, feltParams);
    if (feltContour.empty()) {
        if (outProcessingImage) outProcessingImage->release();
        return;
    }

    cv::Mat railMask = detectRailMask(src, feltContour, railParams);
    if (railMask.empty() || cv::countNonZero(railMask) == 0) {
        if (outProcessingImage) outProcessingImage->release();
        return;
    }

    // Step 2: Apply color filter (always enabled)
    //
    // Rationale:
    // - The UI toggle to disable filtering has been removed. We always use the HSV range.
    // - Defaults are wide enough (full HSV ranges) that this is a no-op until the user picks a color.
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Create a mask for pixels matching the HSV range.
    //
    // Hue wrapping:
    // - OpenCV Hue is [0..180]. If the picked hue is near 0 (or 180) and the tolerance spans across
    //   the boundary, we represent that in params by allowing `colorHMin > colorHMax`.
    // - In that case, we create two ranges and OR them together.
    cv::Mat colorMask;
    if (diamondParams.colorHMin <= diamondParams.colorHMax) {
        cv::inRange(
            hsv,
            cv::Scalar(diamondParams.colorHMin, diamondParams.colorSMin, diamondParams.colorVMin),
            cv::Scalar(diamondParams.colorHMax, diamondParams.colorSMax, diamondParams.colorVMax),
            colorMask);
    } else {
        // Wrapped hue interval: [0..HMax] U [HMin..180]
        cv::Mat lowMask;
        cv::Mat highMask;
        cv::inRange(
            hsv,
            cv::Scalar(0, diamondParams.colorSMin, diamondParams.colorVMin),
            cv::Scalar(diamondParams.colorHMax, diamondParams.colorSMax, diamondParams.colorVMax),
            lowMask);
        cv::inRange(
            hsv,
            cv::Scalar(diamondParams.colorHMin, diamondParams.colorSMin, diamondParams.colorVMin),
            cv::Scalar(180, diamondParams.colorSMax, diamondParams.colorVMax),
            highMask);
        cv::bitwise_or(lowMask, highMask, colorMask);
    }

    // Step 3: Combine spatial mask (rails only) and color mask, then isolate the rail region.
    // This avoids doing two separate copies and ensures we only process relevant pixels.
    cv::Mat combinedMask;
    cv::bitwise_and(colorMask, railMask, combinedMask);

    cv::Mat rail_only = cv::Mat::zeros(src.size(), src.type());
    src.copyTo(rail_only, combinedMask);

    // Step 4: Convert to grayscale
    cv::Mat gray;
    if (rail_only.channels() == 3) {
        cv::cvtColor(rail_only, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = rail_only.clone();
    }

    // Step 5: Apply morphological operations to enhance diamond contrast (optional)
    // Step 6: Threshold to find diamonds
    cv::Mat thresh;
    cv::Mat enhanced;  // For debug output
    cv::Mat tophat, blackhat;  // For debug output (only used if morph enhancement enabled)
    
    if (diamondParams.skip_morph_enhancement) {
        // Skip morphological enhancement - threshold grayscale directly
        enhanced = gray.clone();
        
        // Use adaptive thresholding on grayscale
        int blockSize = diamondParams.adaptive_thresh_blocksize;
        if (blockSize < 3) blockSize = 11;
        if (blockSize % 2 == 0) blockSize++;  // Must be odd
        
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, diamondParams.adaptive_thresh_C);
        
        // Exclude masked-out black areas
        cv::Mat rail_region_valid = gray > 0;
        cv::bitwise_and(thresh, rail_region_valid, thresh);
    } else {
        // Original approach: morphological enhancement then threshold
        // Diamonds are small circular features - use top-hat/black-hat to enhance them
        
        // Get morphological kernel size (should be larger than diamond size)
        int morphSize = diamondParams.morph_kernel_size;
        if (morphSize < 3) morphSize = 15;  // Default to 15 if invalid
        if (morphSize % 2 == 0) morphSize++;  // Ensure odd
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morphSize, morphSize));
        
        // Compute both top-hat and black-hat to enhance both bright and dark features
        // Top-hat: highlights bright spots (diamonds brighter than rail)
        cv::Mat opened;
        cv::morphologyEx(gray, opened, cv::MORPH_OPEN, kernel);
        cv::subtract(gray, opened, tophat);

        // Black-hat: highlights dark spots (diamonds darker than rail)
        cv::Mat closed;
        cv::morphologyEx(gray, closed, cv::MORPH_CLOSE, kernel);
        cv::subtract(closed, gray, blackhat);

        // Combine both results - take maximum of top-hat and black-hat
        // This enhances diamonds regardless of whether they're brighter or darker
        cv::max(tophat, blackhat, enhanced);

        // Threshold the enhanced image
        int threshold = (diamondParams.min_threshold > 0) ? diamondParams.min_threshold : diamondParams.threshold1;
        threshold = std::clamp(threshold, 0, 255);
        
        cv::threshold(enhanced, thresh, threshold, 255, cv::THRESH_BINARY);
        
        // Exclude masked-out black areas (value=0 in original gray) that aren't real spots
        cv::Mat rail_region_valid = gray > 0;  // pixels that have actual data (not masked out)
        cv::bitwise_and(thresh, rail_region_valid, thresh);
    }

    // Step 6: Morphological cleanup (opening: erode then dilate to remove noise)
    // Use a small kernel for cleanup (separate from the large morphological enhancement kernel)
    cv::Mat cleanupKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(thresh, thresh, cleanupKernel, cv::Point(-1, -1), 1);
    cv::dilate(thresh, thresh, cleanupKernel, cv::Point(-1, -1), 1);

    // Step 7: Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Step 8: Filter contours and extract centroids
    std::vector<cv::Point2f> allDiamonds;
    
    // Get area and circularity thresholds - use more permissive values to catch all diamonds
    int min_area = (diamondParams.minArea > 0) ? diamondParams.minArea : 10;  // Lowered from 20
    int max_area = (diamondParams.maxArea > 0) ? diamondParams.maxArea : 500;  // Increased from 300
    float min_circularity = (diamondParams.min_circularity > 0.0f) ? diamondParams.min_circularity : 0.3f;  // Lowered from 0.4

    // Debug: Track rejected contours for diagnostics
    std::vector<std::pair<cv::Point2f, std::string>> rejectedContours;  // centroid, rejection reason
    
    for (const auto& contour : contours) {
        // Calculate area and circularity first
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        double circularity = (perimeter > 0) ? (4.0 * CV_PI * area) / (perimeter * perimeter) : 0.0;
        
        // Extract centroid for diagnostic purposes
        cv::Moments M = cv::moments(contour);
        cv::Point2f centroid(0, 0);
        if (M.m00 > 0) {
            centroid.x = static_cast<float>(M.m10 / M.m00);
            centroid.y = static_cast<float>(M.m01 / M.m00);
        }
        
        // Area filter
        if (area < min_area) {
            rejectedContours.push_back({centroid, "Area too small: " + std::to_string((int)area) + " < " + std::to_string(min_area)});
            continue;
        }
        if (area > max_area) {
            rejectedContours.push_back({centroid, "Area too large: " + std::to_string((int)area) + " > " + std::to_string(max_area)});
            continue;
        }

        // Circularity filter
        if (perimeter == 0) {
            rejectedContours.push_back({centroid, "Zero perimeter"});
            continue;
        }
        if (circularity < min_circularity) {
            rejectedContours.push_back({centroid, "Circularity too low: " + std::to_string(circularity) + " < " + std::to_string(min_circularity)});
            continue;
        }

        // Passed all filters - add to diamonds
        allDiamonds.push_back(centroid);
    }

    // Step 8.5: Remove duplicate detections (diamonds detected multiple times)
    std::vector<cv::Point2f> diamonds;
    
    // Sort by X coordinate for easier duplicate removal
    std::sort(allDiamonds.begin(), allDiamonds.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (std::abs(a.y - b.y) < 5.0f) {  // If roughly same Y (within 5 pixels)
            return a.x < b.x;  // Sort by X
        }
        return a.y < b.y;  // Otherwise sort by Y
    });
    
    // Remove duplicates - if two diamonds are very close, keep only one
    const float duplicateThreshold = 15.0f;  // pixels
    for (const auto& pt : allDiamonds) {
        bool isDuplicate = false;
        for (const auto& existing : diamonds) {
            float dx = pt.x - existing.x;
            float dy = pt.y - existing.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < duplicateThreshold) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            diamonds.push_back(pt);
        }
    }

    // Step 9: Draw detected diamonds on the overlay (with geometric validation)
    if (!diamonds.empty()) {
        cv::Mat overlay = dst.clone();
        
        const int marker_radius = std::max(2, diamondParams.radiusPx);
        const cv::Scalar fillColor = diamondParams.color;
        const cv::Scalar outlineColor = cv::Scalar(0, 0, 0); // Black outline
        const cv::Scalar falsePositiveColor = cv::Scalar(140, 140, 140); // Grey
        

        // -----------------------------------------------------------------------------------------
        // Geometric validation rules (goal-state):
        // 1) 18 Diamonds Total
        // 2) 3 on top short rail
        // 3) 3 on bottom short rail
        // 4) 6 on left long rail (3, pocket, 3)
        // 5) 6 on right long rail (3, pocket, 3)
        //
        // Approach (this step):
        // - Approximate the felt boundary with a rotated rectangle (minAreaRect over felt contour).
        // - Treat its 4 edges as "rail centerlines" and assign each detected point to its nearest edge line.
        // - Determine which two edges are short vs long by edge length.
        // - For each edge, select the best 3 or 6 points using:
        //     - closeness to the edge line
        //     - spacing consistency along the edge (including a central pocket gap on long rails)
        // - Draw non-selected points in grey (false positives) rather than hiding them.
        // -----------------------------------------------------------------------------------------
        cv::RotatedRect feltRect = cv::minAreaRect(feltContour);
        std::array<cv::Point2f, 4> rectPts{};
        feltRect.points(rectPts.data());

        struct Edge { cv::Point2f a; cv::Point2f b; cv::Point2f mid; float len = 0.0f; };
        std::array<Edge, 4> edges{};
        for (int i = 0; i < 4; i++) {
            const cv::Point2f a = rectPts[i];
            const cv::Point2f b = rectPts[(i + 1) % 4];
            const cv::Point2f ab = b - a;
            edges[i] = Edge{a, b, (a + b) * 0.5f, std::sqrt(ab.dot(ab))};
        }

        // Assign each point to nearest edge line.
        std::array<std::vector<int>, 4> assigned;
        for (int pi = 0; pi < (int)diamonds.size(); pi++) {
            const cv::Point2f p = diamonds[pi];
            float bestD = std::numeric_limits<float>::infinity();
            int bestEdge = 0;
            for (int ei = 0; ei < 4; ei++) {
                const float d = distancePointToInfiniteLine(p, edges[ei].a, edges[ei].b);
                if (d < bestD) { bestD = d; bestEdge = ei; }
            }
            assigned[bestEdge].push_back(pi);
        }

        // Determine which opposite edge pair is "long rail" (6) vs "short rail" (3).
        //
        // Why?
        // - Under perspective projection, pixel lengths of the rectangle edges are not reliable
        //   for deciding which physical rail is longer.
        // - Instead, we infer it from the expected diamond spacing pattern:
        //   - long rail: 6 markers with a prominent pocket gap between 3 and 3
        //   - short rail: 3 markers roughly equally spaced
        //
        // Edges are consecutive; opposite pairs are (0,2) and (1,3).
        std::array<RailSubsetResult, 4> best3{};
        std::array<RailSubsetResult, 4> best6{};
        for (int ei = 0; ei < 4; ei++) {
            best3[ei] = chooseBestRailSubset(assigned[ei], diamonds, edges[ei].a, edges[ei].b, 3);
            best6[ei] = chooseBestRailSubset(assigned[ei], diamonds, edges[ei].a, edges[ei].b, 6);
        }

        const float scoreA = best6[0].score + best6[2].score + best3[1].score + best3[3].score; // (0,2) long
        const float scoreB = best3[0].score + best3[2].score + best6[1].score + best6[3].score; // (1,3) long

        std::array<int, 4> requiredPerEdge{3, 3, 3, 3};
        if (scoreA <= scoreB) {
            requiredPerEdge[0] = 6;
            requiredPerEdge[2] = 6;
        } else {
            requiredPerEdge[1] = 6;
            requiredPerEdge[3] = 6;
        }

        // Select best points per edge using the inferred counts.
        std::array<std::vector<int>, 4> acceptedByEdge;
        std::vector<bool> isAccepted(diamonds.size(), false);
        for (int ei = 0; ei < 4; ei++) {
            const int k = requiredPerEdge[ei];
            RailSubsetResult r = chooseBestRailSubset(assigned[ei], diamonds, edges[ei].a, edges[ei].b, k);
            acceptedByEdge[ei] = std::move(r.indices);
            for (int idx : acceptedByEdge[ei]) {
                if (idx >= 0 && idx < (int)isAccepted.size()) isAccepted[idx] = true;
            }
        }

        // Draw diamonds: accepted use configured color; non-accepted use grey (false positives).
        for (int i = 0; i < (int)diamonds.size(); i++) {
            const cv::Point2f pt = diamonds[i];
            cv::Point center((int)std::lround(pt.x), (int)std::lround(pt.y));

            const cv::Scalar c = isAccepted[i] ? fillColor : falsePositiveColor;
            const int thickness = isAccepted[i] ? diamondParams.outlineThicknessPx : std::max(1, diamondParams.outlineThicknessPx - 1);

            if (diamondParams.isFilled) {
                cv::circle(overlay, center, marker_radius, c, -1);
            } else {
                cv::circle(overlay, center, marker_radius, c, thickness);
            }
            cv::circle(overlay, center, marker_radius, outlineColor, thickness);
        }

        // -----------------------------------------------------------------------------------------
        // TEMPORARY DEBUG: draw best-fit rail lines and counts.
        // -----------------------------------------------------------------------------------------
        const cv::Scalar orange = cv::Scalar(0, 165, 255); // BGR
        const cv::Scalar white = cv::Scalar(255, 255, 255);

        int totalAccepted = 0;
        for (int ei = 0; ei < 4; ei++) {
            totalAccepted += (int)acceptedByEdge[ei].size();
        }

        for (int ei = 0; ei < 4; ei++) {
            // Build point list for line drawing (prefer accepted points; fallback to assigned if needed).
            std::vector<cv::Point2f> ptsForLine;
            ptsForLine.reserve(acceptedByEdge[ei].size());
            for (int idx : acceptedByEdge[ei]) ptsForLine.push_back(diamonds[idx]);
            if ((int)ptsForLine.size() < 2) {
                ptsForLine.clear();
                for (int idx : assigned[ei]) ptsForLine.push_back(diamonds[idx]);
            }

            // Draw only if we can anchor through at least 2 diamond centers.
            // This avoids misleading lines that visually miss all diamonds.
            (void)drawAnchoredRailLineFromPoints(overlay, ptsForLine, orange, 2);

            // Count label near edge mid-point.
            const int have = (int)acceptedByEdge[ei].size();
            const int want = requiredPerEdge[ei];
            const std::string label = std::to_string(have) + "/" + std::to_string(want);

            cv::Point pos((int)std::lround(edges[ei].mid.x), (int)std::lround(edges[ei].mid.y));
            pos.x = std::clamp(pos.x, 10, overlay.cols - 80);
            pos.y = std::clamp(pos.y, 20, overlay.rows - 10);

            cv::putText(overlay, label, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, outlineColor, 3, cv::LINE_AA);
            cv::putText(overlay, label, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1, cv::LINE_AA);
        }

        // Total count label (18 is expected).
        {
            const std::string totalLabel = "Diamonds: " + std::to_string(totalAccepted) + "/18";
            cv::Point pos(15, 25);
            cv::putText(overlay, totalLabel, pos, cv::FONT_HERSHEY_SIMPLEX, 0.7, outlineColor, 3, cv::LINE_AA);
            cv::putText(overlay, totalLabel, pos, cv::FONT_HERSHEY_SIMPLEX, 0.7, white, 1, cv::LINE_AA);
        }

        // Apply alpha blending
        const double alpha = std::clamp(diamondParams.alpha, 0, 255) / 255.0;
        cv::addWeighted(dst, 1.0 - alpha, overlay, alpha, 0, dst);
    }

    // Set processing image output (the thresholded image)
    if (outProcessingImage) {
        *outProcessingImage = thresh.clone();
    }

    // Add debug images
    g_lastDiamondDebugImages.push_back(std::make_pair("Rail Region", rail_only));
    g_lastDiamondDebugImages.push_back(std::make_pair("Grayscale", gray));
    if (!diamondParams.skip_morph_enhancement) {
        // Only show top-hat/black-hat if we did morphological enhancement
        g_lastDiamondDebugImages.push_back(std::make_pair("Top-hat", tophat));
        g_lastDiamondDebugImages.push_back(std::make_pair("Black-hat", blackhat));
    }
    g_lastDiamondDebugImages.push_back(std::make_pair("Enhanced", enhanced));
    g_lastDiamondDebugImages.push_back(std::make_pair("Threshold", thresh));
    
    // Debug: Show all contours and rejected ones
    cv::Mat contourDebug = src.clone();
    // Draw all contours found (before filtering) in yellow
    cv::drawContours(contourDebug, contours, -1, cv::Scalar(0, 255, 255), 1);
    // Draw rejected contours in magenta
    for (const auto& rejected : rejectedContours) {
        cv::circle(contourDebug, cv::Point(static_cast<int>(rejected.first.x), static_cast<int>(rejected.first.y)), 
                   5, cv::Scalar(255, 0, 255), 2);
    }
    // Draw accepted diamonds in green
    for (const auto& pt : allDiamonds) {
        cv::circle(contourDebug, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 
                   5, cv::Scalar(0, 255, 0), 2);
    }
    g_lastDiamondDebugImages.push_back(std::make_pair("Contour Debug (Yellow=all, Magenta=rejected, Green=accepted)", contourDebug));
    
}
