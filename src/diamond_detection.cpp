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
    const RailParams& railParams,
    DiamondDebugImages* debugOut)
{
    if (src.empty()) return;
    if (dst.empty()) dst = src.clone();

    if (debugOut) {
        debugOut->railMaskDark = cv::Mat::zeros(src.size(), CV_8UC1);
        debugOut->railSearchMask = cv::Mat::zeros(src.size(), CV_8UC1);
        debugOut->railEnhanced = cv::Mat::zeros(src.size(), CV_8UC1);
        debugOut->otsuBinary = cv::Mat::zeros(src.size(), CV_8UC1);
        debugOut->diamondMask = cv::Mat::zeros(src.size(), CV_8UC1);
        debugOut->roi = cv::Rect();
        debugOut->keypointsFound = 0;
        debugOut->centersKept = 0;
    }
}
