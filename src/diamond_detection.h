#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

// We accept these parameter structs so the sidebar tuning actually impacts diamond detection.
#include "felt_detection.h"
#include "rail_detection.h"

// Structure to hold detection parameters
struct DiamondDetectionParams {
    int threshold1 = 0;
    int threshold2 = 255;
    int minArea = 5;
    int maxArea = 1000;
    // Overlay styling
    cv::Scalar color = cv::Scalar(0, 255, 255); // BGR
    // Alpha (transparency) for rendering: 0..255 (0 = invisible, 255 = fully opaque)
    int alpha = 255;
    bool isFilled = true;
    int radiusPx = 8;
    int outlineThicknessPx = 2;
};

// Debug images emitted by the diamond detection pipeline (label, image)
extern std::vector<std::pair<std::string, cv::Mat>> g_lastDiamondDebugImages;

// Detect diamond markers on the pool table rails with context-aware constraints
void detectDiamonds(
    const cv::Mat& src,
    cv::Mat& dst,
    bool showDiamonds,
    const DiamondDetectionParams& diamondParams,
    const FeltParams& feltParams,
    const RailParams& railParams,
    cv::Mat* outProcessingImage = nullptr);
