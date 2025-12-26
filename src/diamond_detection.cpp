#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"

#include <opencv2/imgproc.hpp>

void detectDiamonds(
    const cv::Mat& src,
    cv::Mat& dst,
    bool showDiamonds,
    const DiamondDetectionParams& diamondParams,
    const FeltParams& feltParams,
    const RailParams& railParams)
{
    if (src.empty()) return;
    if (dst.empty()) dst = src.clone();
}
