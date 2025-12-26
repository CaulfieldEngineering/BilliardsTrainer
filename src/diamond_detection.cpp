#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"

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

    // Stub implementation - no-op
    if (outProcessingImage) {
        outProcessingImage->release();
    }

    // Mark unused parameters to avoid compiler warnings
    (void)diamondParams;
    (void)feltParams;
    (void)railParams;
}
