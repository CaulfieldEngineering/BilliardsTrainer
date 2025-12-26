#include "orientation_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

// -----------------------------
// Table orientation (Hough + clustering)
// -----------------------------

TableOrientation computeTableOrientationFromRailMask(const cv::Mat& railMask) {
    TableOrientation result;
    if (railMask.empty() || cv::countNonZero(railMask) < 100) return result;

    // 1) Clean the mask: morphological close then open.
    cv::Mat cleaned;
    {
        const int ksize = 5;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
        cv::morphologyEx(railMask, cleaned, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
    }

    // Keep largest connected component.
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return result;
        int bestIdx = 0;
        double bestArea = 0;
        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > bestArea) { bestArea = a; bestIdx = i; }
        }
        cleaned = cv::Mat::zeros(railMask.size(), CV_8UC1);
        cv::drawContours(cleaned, contours, bestIdx, cv::Scalar(255), cv::FILLED);
    }

    // 2) Extract edges using Canny.
    cv::Mat edges;
    cv::Canny(cleaned, edges, 50, 150);

    // 3) Hough lines to find dominant orientations.
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);
    if (lines.size() < 4) return result;

    // Collect angles (folded to [0, pi)).
    std::vector<double> angles;
    angles.reserve(lines.size());
    for (const auto& l : lines) {
        const double dx = l[2] - l[0];
        const double dy = l[3] - l[1];
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len < 20) continue; // Skip short segments.
        double theta = std::atan2(dy, dx);
        theta = std::fmod(theta + CV_PI, CV_PI); // fold
        angles.push_back(theta);
    }
    if (angles.size() < 4) return result;

    // 4) Cluster angles into two families using simple k-means on circular space [0, pi).
    // Initialize centers as the two most separated angles.
    double centerA = angles[0];
    double centerB = angles[0];
    double maxSep = 0;
    for (size_t i = 0; i < angles.size(); i++) {
        for (size_t j = i + 1; j < angles.size(); j++) {
            double diff = std::abs(angles[i] - angles[j]);
            if (diff > CV_PI / 2) diff = CV_PI - diff; // circular distance on [0, pi)
            if (diff > maxSep) { maxSep = diff; centerA = angles[i]; centerB = angles[j]; }
        }
    }

    auto circularMean = [](const std::vector<double>& vals) -> double {
        // Double-angle trick to handle pi-periodicity.
        double sx = 0.0, sy = 0.0;
        for (double v : vals) {
            sx += std::cos(2.0 * v);
            sy += std::sin(2.0 * v);
        }
        double mean2 = std::atan2(sy, sx);
        double mean = mean2 / 2.0;
        if (mean < 0) mean += CV_PI;
        return mean;
    };

    for (int iter = 0; iter < 10; iter++) {
        std::vector<double> groupA, groupB;
        for (double a : angles) {
            double dA = std::abs(a - centerA);
            if (dA > CV_PI / 2) dA = CV_PI - dA;
            double dB = std::abs(a - centerB);
            if (dB > CV_PI / 2) dB = CV_PI - dB;
            if (dA < dB) groupA.push_back(a);
            else groupB.push_back(a);
        }
        if (groupA.empty() || groupB.empty()) break;
        centerA = circularMean(groupA);
        centerB = circularMean(groupB);
    }

    // 5) Stabilize sign: choose a consistent direction (dir.x >= 0).
    result.thetaA = centerA;
    result.thetaB = centerB;
    result.dirA = cv::Point2f((float)std::cos(centerA), (float)std::sin(centerA));
    result.dirB = cv::Point2f((float)std::cos(centerB), (float)std::sin(centerB));

    if (result.dirA.x < 0) {
        result.dirA *= -1;
        result.thetaA = std::fmod(result.thetaA + CV_PI, CV_PI);
    }
    if (result.dirB.x < 0) {
        result.dirB *= -1;
        result.thetaB = std::fmod(result.thetaB + CV_PI, CV_PI);
    }

    result.valid = true;
    return result;
}

// -----------------------------
// Orientation Mask overlay helpers
// -----------------------------

namespace {

struct LineNF {
    // Normal form: n·x + c = 0 (n is unit-length)
    cv::Point2f n;
    float c = 0.0f;
};

static bool intersectLines(const LineNF& a, const LineNF& b, cv::Point2f& out) {
    const float A00 = a.n.x;
    const float A01 = a.n.y;
    const float A10 = b.n.x;
    const float A11 = b.n.y;
    const float B0 = -a.c;
    const float B1 = -b.c;
    const float det = A00 * A11 - A01 * A10;
    if (std::abs(det) < 1e-6f) return false;
    out.x = (B0 * A11 - A01 * B1) / det;
    out.y = (A00 * B1 - B0 * A10) / det;
    return std::isfinite(out.x) && std::isfinite(out.y);
}

static float distPointToInfLine(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f ab = b - a;
    const float abLen = std::sqrt(ab.dot(ab));
    if (abLen <= 1e-6f) {
        const cv::Point2f d = p - a;
        return std::sqrt(d.dot(d));
    }
    const cv::Point2f ap = p - a;
    const float cross = std::abs(ab.x * ap.y - ab.y * ap.x);
    return cross / abLen;
}

// Derive a 4-corner quad from a contour.
// If shouldEncloseHull is true, post-process into an enclosing quad by pushing edges outward to contain the full hull.
static bool quadFromContour(const std::vector<cv::Point>& contour, cv::Point2f outPts[4], bool shouldEncloseHull) {
    if (contour.size() < 4) return false;

    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    if (hull.size() < 4) return false;

    const double peri = std::max(1.0, cv::arcLength(hull, true));
    std::vector<cv::Point> approx;

    bool found = false;
    for (double k = 0.01; k <= 0.10; k += 0.01) {
        approx.clear();
        cv::approxPolyDP(hull, approx, k * peri, true);
        if (approx.size() == 4) { found = true; break; }
        if (approx.size() < 4) break;
    }

    if (!found) {
        cv::RotatedRect rr = cv::minAreaRect(hull);
        rr.points(outPts);
    } else {
        cv::Point2f c(0, 0);
        for (const auto& p : approx) c += cv::Point2f((float)p.x, (float)p.y);
        c *= 0.25f;

        std::sort(approx.begin(), approx.end(), [&](const cv::Point& a, const cv::Point& b) {
            const float aa = std::atan2((float)a.y - c.y, (float)a.x - c.x);
            const float bb = std::atan2((float)b.y - c.y, (float)b.x - c.x);
            return aa < bb;
        });

        for (int i = 0; i < 4; i++) outPts[i] = cv::Point2f((float)approx[i].x, (float)approx[i].y);

        if (shouldEncloseHull) {
            // Expand the quad to enclose ALL hull points by shifting each edge line outward.
            // We treat each edge as a half-plane and move it outward by the maximum violation.
            struct Line { cv::Point2f n; float c; };
            Line lines[4]{};

            cv::Point2f qc(0, 0);
            for (int i = 0; i < 4; i++) qc += outPts[i];
            qc *= 0.25f;

            for (int i = 0; i < 4; i++) {
                const cv::Point2f p0 = outPts[i];
                const cv::Point2f p1 = outPts[(i + 1) % 4];
                const cv::Point2f d = p1 - p0;
                float len = std::sqrt(d.dot(d));
                if (len < 1e-3f) len = 1.0f;

                cv::Point2f n(d.y / len, -d.x / len);
                float cc = -(n.dot(p0));

                if (n.dot(qc) + cc > 0.0f) { n *= -1.0f; cc *= -1.0f; }

                float maxV = 0.0f;
                for (const auto& hp : hull) {
                    const cv::Point2f pf((float)hp.x, (float)hp.y);
                    const float v = n.dot(pf) + cc;
                    if (v > maxV) maxV = v;
                }

                cc -= maxV;
                lines[i] = Line{n, cc};
            }

            auto intersect = [&](const Line& a, const Line& b, cv::Point2f& out) -> bool {
                const float A00 = a.n.x;
                const float A01 = a.n.y;
                const float A10 = b.n.x;
                const float A11 = b.n.y;
                const float B0 = -a.c;
                const float B1 = -b.c;
                const float det = A00 * A11 - A01 * A10;
                if (std::abs(det) < 1e-6f) return false;
                out.x = (B0 * A11 - A01 * B1) / det;
                out.y = (A00 * B1 - B0 * A10) / det;
                return true;
            };

            cv::Point2f newPts[4];
            bool ok = true;
            for (int i = 0; i < 4; i++) {
                if (!intersect(lines[i], lines[(i + 1) % 4], newPts[(i + 1) % 4])) { ok = false; break; }
            }
            if (ok) for (int i = 0; i < 4; i++) outPts[i] = newPts[i];
        }
    }

    return true;
}

static void orderQuadTLTRBRBL(cv::Point2f pts[4]) {
    // Using sum/diff heuristic:
    // - TL: min(x+y), BR: max(x+y)
    // - TR: min(x-y), BL: max(x-y)
    cv::Point2f ordered[4];
    int tl = 0, tr = 0, br = 0, bl = 0;
    float minSum = std::numeric_limits<float>::infinity();
    float maxSum = -std::numeric_limits<float>::infinity();
    float minDiff = std::numeric_limits<float>::infinity();
    float maxDiff = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; i++) {
        const float s = pts[i].x + pts[i].y;
        const float d = pts[i].x - pts[i].y;
        if (s < minSum) { minSum = s; tl = i; }
        if (s > maxSum) { maxSum = s; br = i; }
        if (d < minDiff) { minDiff = d; tr = i; }
        if (d > maxDiff) { maxDiff = d; bl = i; }
    }
    ordered[0] = pts[tl]; // TL
    ordered[1] = pts[tr]; // TR
    ordered[2] = pts[br]; // BR
    ordered[3] = pts[bl]; // BL
    for (int i = 0; i < 4; i++) pts[i] = ordered[i];
}

static void reorderInnerToMatchOuter(const cv::Point2f outerPts[4], cv::Point2f innerPts[4]) {
    cv::Point2f orderedInner[4];
    bool used[4] = {false, false, false, false};
    for (int i = 0; i < 4; i++) {
        float bestD = std::numeric_limits<float>::infinity();
        int bestJ = 0;
        for (int j = 0; j < 4; j++) {
            if (used[j]) continue;
            const cv::Point2f d = innerPts[j] - outerPts[i];
            const float dd = d.dot(d);
            if (dd < bestD) { bestD = dd; bestJ = j; }
        }
        used[bestJ] = true;
        orderedInner[i] = innerPts[bestJ];
    }
    for (int i = 0; i < 4; i++) innerPts[i] = orderedInner[i];
}

static void smooth1D(std::vector<double>& v, int radius) {
    if (v.empty() || radius <= 0) return;
    std::vector<double> out(v.size(), 0.0);
    for (int i = 0; i < (int)v.size(); i++) {
        const int a = std::max(0, i - radius);
        const int b = std::min((int)v.size() - 1, i + radius);
        double s = 0.0;
        for (int j = a; j <= b; j++) s += v[j];
        out[i] = s / (double)(b - a + 1);
    }
    v.swap(out);
}

} // namespace

void drawOrientationMaskOverlay(
    cv::Mat& frameBGR,
    const cv::Mat& railMask,
    const std::vector<cv::Point>& feltContour,
    const OrientationMaskRenderParams& params) {

    if (frameBGR.empty()) return;
    if (railMask.empty() || cv::countNonZero(railMask) <= 0) return;
    if (feltContour.empty()) return;

    // Outer quad: from rail mask boundary.
    std::vector<std::vector<cv::Point>> railContours;
    cv::findContours(railMask, railContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (railContours.empty()) return;

    size_t bestIdx = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < railContours.size(); i++) {
        const double area = cv::contourArea(railContours[i]);
        if (area > bestArea) { bestArea = area; bestIdx = i; }
    }

    cv::Point2f outerPts[4];
    cv::Point2f innerPts[4];
    if (!quadFromContour(railContours[bestIdx], outerPts, /*shouldEncloseHull*/true)) return;

    // Inner quad: fit 4 edge lines directly to felt contour using a perspective-following hull quad for bucketing.
    bool innerOk = false;
    {
        // Get felt's perspective-following quad for bucketing guidance.
        cv::Point2f feltQuadPts[4];
        {
            std::vector<cv::Point> feltHull;
            cv::convexHull(feltContour, feltHull);
            double peri = cv::arcLength(feltHull, true);
            std::vector<cv::Point> approx;
            for (double eps = 0.01; eps < 0.15; eps += 0.005) {
                cv::approxPolyDP(feltHull, approx, eps * peri, true);
                if (approx.size() <= 4) break;
            }
            if (approx.size() == 4) {
                cv::Point2f ctr(0, 0);
                for (const auto& p : approx) ctr += cv::Point2f((float)p.x, (float)p.y);
                ctr *= 0.25f;
                std::vector<std::pair<float, int>> angles(4);
                for (int i = 0; i < 4; i++) {
                    float dx = approx[i].x - ctr.x;
                    float dy = approx[i].y - ctr.y;
                    angles[i] = {std::atan2(dy, dx), i};
                }
                std::sort(angles.begin(), angles.end());
                for (int i = 0; i < 4; i++) {
                    feltQuadPts[i] = cv::Point2f((float)approx[angles[i].second].x,
                                                 (float)approx[angles[i].second].y);
                }
            } else {
                cv::RotatedRect rr = cv::minAreaRect(feltContour);
                rr.points(feltQuadPts);
            }
        }

        // Bucket points by nearest felt-quad edge.
        std::vector<cv::Point2f> sidePts[4];
        for (int i = 0; i < 4; i++) sidePts[i].reserve(feltContour.size() / 4);
        for (const auto& pI : feltContour) {
            const cv::Point2f p((float)pI.x, (float)pI.y);
            float bestD = std::numeric_limits<float>::infinity();
            int bestEdge = 0;
            for (int i = 0; i < 4; i++) {
                const float d = distPointToInfLine(p, feltQuadPts[i], feltQuadPts[(i + 1) % 4]);
                if (d < bestD) { bestD = d; bestEdge = i; }
            }
            sidePts[bestEdge].push_back(p);
        }

        // Fit one line per side (in normal form).
        LineNF lines[4];
        bool ok = true;
        for (int i = 0; i < 4; i++) {
            if ((int)sidePts[i].size() < 20) { ok = false; break; }
            cv::Vec4f lf;
            cv::fitLine(sidePts[i], lf, cv::DIST_L2, 0, 0.01, 0.01);
            const float vx = lf[0], vy = lf[1], x0 = lf[2], y0 = lf[3];
            cv::Point2f n(vy, -vx);
            const float nLen = std::sqrt(n.dot(n));
            if (nLen <= 1e-6f) { ok = false; break; }
            n *= (1.0f / nLen);
            const float c = -(n.x * x0 + n.y * y0);
            lines[i] = LineNF{n, c};
        }

        if (ok) {
            cv::Point2f pts[4];
            for (int i = 0; i < 4; i++) {
                const LineNF& prev = lines[(i + 3) % 4];
                const LineNF& cur = lines[i];
                if (!intersectLines(prev, cur, pts[i])) { ok = false; break; }
            }
            if (ok) {
                for (int i = 0; i < 4; i++) innerPts[i] = pts[i];
                innerOk = true;
            }
        }

        if (!innerOk) {
            // Fallback: simpler quad (less tight).
            (void)quadFromContour(feltContour, innerPts, /*shouldEncloseHull*/true);
        }
    }

    // Shrink inner quad inward for margin.
    {
        cv::Point2f c(0, 0);
        for (int i = 0; i < 4; i++) c += innerPts[i];
        c *= 0.25f;
        const float shrink = std::clamp(params.innerShrinkFraction, 0.0f, 0.25f);
        for (int i = 0; i < 4; i++) innerPts[i] = innerPts[i] + shrink * (c - innerPts[i]);
    }

    // Stabilize corner ordering and ensure inner/outer correspondence.
    orderQuadTLTRBRBL(outerPts);
    reorderInnerToMatchOuter(outerPts, innerPts);

    auto drawQuad = [&](const cv::Point2f pts[4], const cv::Scalar& color, int thicknessPx) {
        for (int i = 0; i < 4; i++) {
            const cv::Point2f a = pts[i];
            const cv::Point2f b = pts[(i + 1) % 4];
            cv::line(
                frameBGR,
                cv::Point((int)std::lround(a.x), (int)std::lround(a.y)),
                cv::Point((int)std::lround(b.x), (int)std::lround(b.y)),
                color,
                thicknessPx,
                cv::LINE_AA);
        }
    };

    // Dividers: from inner corner to outer corner.
    for (int i = 0; i < 4; i++) {
        cv::line(
            frameBGR,
            cv::Point((int)std::lround(innerPts[i].x), (int)std::lround(innerPts[i].y)),
            cv::Point((int)std::lround(outerPts[i].x), (int)std::lround(outerPts[i].y)),
            params.lineColorBGR,
            params.lineThicknessPx,
            cv::LINE_AA);
    }

    // Regions and mid-pocket notch detection.
    struct RailRegion {
        cv::Point2f innerA, innerB, outerA, outerB;
        cv::Point2f centroid;
        double notchScore = std::numeric_limits<double>::infinity();
        const char* label = "?";
    };
    std::vector<RailRegion> regions(4);
    for (int i = 0; i < 4; i++) {
        const int next = (i + 1) % 4;
        regions[i].innerA = innerPts[i];
        regions[i].innerB = innerPts[next];
        regions[i].outerA = outerPts[i];
        regions[i].outerB = outerPts[next];
        regions[i].centroid = (regions[i].innerA + regions[i].innerB + regions[i].outerA + regions[i].outerB) * 0.25f;
    }

    auto scoreRegionForMiddlePocket = [&](const RailRegion& r) -> double {
        const cv::Point2f e = r.outerB - r.outerA;
        const float eLen = std::sqrt(e.dot(e));
        if (eLen < 5.0f) return std::numeric_limits<double>::infinity();
        const cv::Point2f t = e * (1.0f / eLen);

        // Inward normal: pick sign so it points toward region centroid.
        cv::Point2f n(t.y, -t.x);
        const cv::Point2f mid = (r.outerA + r.outerB) * 0.5f;
        if (n.dot(r.centroid - mid) < 0.0f) n *= -1.0f;

        const int N = 120;
        std::vector<double> prof(N, 0.0);

        for (int i = 0; i < N; i++) {
            const float a = (N == 1) ? 0.5f : (float)i / (float)(N - 1);
            const cv::Point2f pOuter = r.outerA + a * (r.outerB - r.outerA);
            const cv::Point2f pInner = r.innerA + a * (r.innerB - r.innerA);

            float depth = std::sqrt((pInner - pOuter).dot(pInner - pOuter));
            depth = std::clamp(depth, 8.0f, 200.0f);
            const int steps = std::clamp((int)std::lround(depth), 12, 80);

            int hits = 0;
            for (int s = 0; s < steps; s++) {
                const float u = ((float)s + 0.5f) / (float)steps;
                const cv::Point2f q = pOuter + n * (u * depth);
                const int x = (int)std::lround(q.x);
                const int y = (int)std::lround(q.y);
                if ((unsigned)x < (unsigned)railMask.cols && (unsigned)y < (unsigned)railMask.rows) {
                    if (railMask.at<unsigned char>(y, x) != 0) hits++;
                }
            }
            prof[i] = (double)hits / (double)steps;
        }

        smooth1D(prof, 4);

        auto meanRange = [&](double a0, double a1) -> double {
            const int i0 = std::clamp((int)std::lround(a0 * (N - 1)), 0, N - 1);
            const int i1 = std::clamp((int)std::lround(a1 * (N - 1)), 0, N - 1);
            if (i1 <= i0) return 0.0;
            double acc = 0.0;
            for (int i = i0; i <= i1; i++) acc += prof[i];
            return acc / (double)(i1 - i0 + 1);
        };
        auto minRange = [&](double a0, double a1) -> double {
            const int i0 = std::clamp((int)std::lround(a0 * (N - 1)), 0, N - 1);
            const int i1 = std::clamp((int)std::lround(a1 * (N - 1)), 0, N - 1);
            if (i1 <= i0) return 1.0;
            double m = 1.0;
            for (int i = i0; i <= i1; i++) m = std::min(m, prof[i]);
            return m;
        };

        const double centerMin = minRange(0.44, 0.56);
        const double shoulderA = meanRange(0.22, 0.34);
        const double shoulderB = meanRange(0.66, 0.78);
        const double denom = std::max(1e-6, 0.5 * (shoulderA + shoulderB));
        return centerMin / denom; // lower => more notch-like
    };

    for (auto& r : regions) r.notchScore = scoreRegionForMiddlePocket(r);

    std::vector<int> idx = {0, 1, 2, 3};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return regions[a].notchScore < regions[b].notchScore; });

    const int Lidx0 = idx[0];
    const int Lidx1 = idx[1];
    const int Sidx0 = idx[2];
    const int Sidx1 = idx[3];

    auto assignPairLabels = [&](int a, int b, const char* labelA, const char* labelB) {
        const cv::Point2f ca = regions[a].centroid;
        const cv::Point2f cb = regions[b].centroid;
        const float dx = std::abs(ca.x - cb.x);
        const float dy = std::abs(ca.y - cb.y);
        const bool separateInX = (dx > dy);
        const bool aFirst = separateInX ? (ca.x < cb.x) : (ca.y < cb.y);
        if (aFirst) { regions[a].label = labelA; regions[b].label = labelB; }
        else { regions[a].label = labelB; regions[b].label = labelA; }
    };

    assignPairLabels(Lidx0, Lidx1, "L1", "L2");
    assignPairLabels(Sidx0, Sidx1, "S1", "S2");

    // Shade regions.
    cv::Mat overlay = frameBGR.clone();
    for (int i = 0; i < 4; i++) {
        std::vector<cv::Point> poly = {
            cv::Point((int)std::lround(regions[i].innerA.x), (int)std::lround(regions[i].innerA.y)),
            cv::Point((int)std::lround(regions[i].innerB.x), (int)std::lround(regions[i].innerB.y)),
            cv::Point((int)std::lround(regions[i].outerB.x), (int)std::lround(regions[i].outerB.y)),
            cv::Point((int)std::lround(regions[i].outerA.x), (int)std::lround(regions[i].outerA.y))
        };
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{poly}, params.fillColorBGR);
    }
    const double a = std::clamp(params.fillAlpha, 0.0, 1.0);
    cv::addWeighted(overlay, a, frameBGR, 1.0 - a, 0, frameBGR);

    // Draw quads on top.
    drawQuad(outerPts, params.lineColorBGR, params.lineThicknessPx);
    drawQuad(innerPts, params.lineColorBGR, params.lineThicknessPx);

    // Labels.
    for (int i = 0; i < 4; i++) {
        const cv::Point2f centroid = regions[i].centroid;
        const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        const double fontScale = params.labelFontScale;
        const int thickness = params.labelThicknessPx;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(regions[i].label, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg((int)(centroid.x - textSize.width / 2), (int)(centroid.y + textSize.height / 2));
        cv::putText(frameBGR, regions[i].label, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
        cv::putText(frameBGR, regions[i].label, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}


