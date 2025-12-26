#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>

// Owning definition for debug outputs (declared extern in the header)
std::vector<std::pair<std::string, cv::Mat>> g_lastDiamondDebugImages;

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

    // -------------------------------------------------------------------------
    // Minimal, parameter-driven placeholder pipeline:
    // 1) Convert to grayscale and run Canny using UI thresholds.
    // 2) Filter contours by area to reject tiny specks.
    // 3) Draw annotated circles where diamonds would be rendered.
    //
    // This keeps the function buildable and responsive to UI controls while
    // leaving room for a more advanced detector later.
    // -------------------------------------------------------------------------
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray,
              edges,
              std::clamp(diamondParams.threshold1, 0, 255),
              std::clamp(diamondParams.threshold2, 0, 255));

    // Provide the processing image (kept in BGR for consistency with exports)
    cv::Mat processingBgr;
    cv::cvtColor(edges, processingBgr, cv::COLOR_GRAY2BGR);
    if (outProcessingImage) {
        *outProcessingImage = processingBgr.clone();
    }
    g_lastDiamondDebugImages.emplace_back("edges", processingBgr);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < diamondParams.minArea || area > diamondParams.maxArea) {
            continue;
        }

        const cv::Moments m = cv::moments(contour);
        if (m.m00 == 0.0) continue;
        const cv::Point center(static_cast<int>(m.m10 / m.m00),
                               static_cast<int>(m.m01 / m.m00));

        cv::Mat overlay = dst.clone();
        const int thickness = diamondParams.isFilled
            ? cv::FILLED
            : std::max(1, diamondParams.outlineThicknessPx);

        cv::circle(overlay, center, std::max(1, diamondParams.radiusPx), diamondParams.color, thickness, cv::LINE_AA);

        // Apply alpha blending so the overlay respects UI transparency.
        const double alpha = std::clamp(diamondParams.alpha, 0, 255) / 255.0;
        cv::addWeighted(dst, 1.0 - alpha, overlay, alpha, 0, dst);
    }

    // Explicitly mark unused parameters for clarity until the full detector is added.
    (void)feltParams;
    (void)railParams;
}
