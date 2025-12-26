#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"
#include <algorithm>
#include <cmath>

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

    // Create a mask for pixels matching the HSV range
    cv::Mat colorMask;
    cv::inRange(
        hsv,
        cv::Scalar(diamondParams.colorHMin, diamondParams.colorSMin, diamondParams.colorVMin),
        cv::Scalar(diamondParams.colorHMax, diamondParams.colorSMax, diamondParams.colorVMax),
        colorMask);

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

    // Step 9: Draw detected diamonds on the overlay
    if (!diamonds.empty()) {
        cv::Mat overlay = dst.clone();
        
        const int marker_radius = std::max(2, diamondParams.radiusPx);
        const cv::Scalar fillColor = diamondParams.color;
        const cv::Scalar outlineColor = cv::Scalar(0, 0, 0); // Black outline
        
        for (const auto& pt : diamonds) {
            cv::Point center(static_cast<int>(pt.x), static_cast<int>(pt.y));
            
            if (diamondParams.isFilled) {
                // Draw filled circle
                cv::circle(overlay, center, marker_radius, fillColor, -1);
            } else {
                // Draw outline only
                cv::circle(overlay, center, marker_radius, fillColor, diamondParams.outlineThicknessPx);
            }
            
            // Draw outline for definition
            cv::circle(overlay, center, marker_radius, outlineColor, diamondParams.outlineThicknessPx);
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
