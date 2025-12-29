#include "main.h"

#include <fstream> // std::ifstream/std::ofstream
#include <cctype>  // std::isspace
#include <cstdio>  // std::sscanf
#include <cstring> // std::memcpy

// Define global UI state (declared as `extern` in `main.h`)
UIControls uiControls{};

// -------------------------------------------------------------------------------------------------
// "Basic memory" (settings persistence)
//
// We intentionally keep this dead-simple:
// - `settings.txt` in the same directory as the executable (or CWD fallback)
// - key=value pairs, one per line
// - unknown keys ignored (forward compatible)
//
// This is NOT meant to be a full config system; it just preserves the user's last-tuned parameters.
// -------------------------------------------------------------------------------------------------

static std::filesystem::path getSettingsFilePath() {
    // Store settings in the same directory as testImage.jpg (project root)
    // This ensures settings are not in the build folder (which is gitignored)
    // and are preserved across builds and crashes.
    // Use the same resolution logic as testImage.jpg: try current directory, then "../../"
    std::filesystem::path basePath = std::filesystem::current_path();
    
    // Check if testImage.jpg exists in current directory
    if (std::filesystem::exists(basePath / "testImage.jpg")) {
        return basePath / "settings.txt";
    }
    
    // Otherwise, use "../../" (assuming we're in build/Release or similar)
    // Go up two directory levels to reach the project root
    std::filesystem::path altPath = basePath.parent_path().parent_path() / "settings.txt";
    return altPath;
}

static std::string trimCopy(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) a++;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static bool parseBool(const std::string& v, bool& out) {
    const std::string s = trimCopy(v);
    if (s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes" || s == "Yes") { out = true; return true; }
    if (s == "0" || s == "false" || s == "False" || s == "FALSE" || s == "no" || s == "No") { out = false; return true; }
    return false;
}

static bool parseInt(const std::string& v, int& out) {
    try {
        size_t idx = 0;
        const int x = std::stoi(trimCopy(v), &idx);
        out = x;
        return true;
    } catch (...) { return false; }
}

static bool parseFloat(const std::string& v, float& out) {
    try {
        size_t idx = 0;
        const float x = std::stof(trimCopy(v), &idx);
        out = x;
        return true;
    } catch (...) { return false; }
}

static std::string toStringBool(bool v) { return v ? "1" : "0"; }

static std::string toStringVec3b(const cv::Vec3b& v) {
    return std::to_string((int)v[0]) + "," + std::to_string((int)v[1]) + "," + std::to_string((int)v[2]);
}

static bool parseVec3b(const std::string& s, cv::Vec3b& out) {
    int a = 0, b = 0, c = 0;
    if (std::sscanf(s.c_str(), "%d,%d,%d", &a, &b, &c) == 3) {
        out = cv::Vec3b((uchar)std::clamp(a, 0, 255), (uchar)std::clamp(b, 0, 255), (uchar)std::clamp(c, 0, 255));
        return true;
    }
    return false;
}

static std::string toStringScalarBgr(const cv::Scalar& bgr) {
    // BGR stored as integers for readability.
    return std::to_string((int)std::lround(bgr[0])) + "," + std::to_string((int)std::lround(bgr[1])) + "," + std::to_string((int)std::lround(bgr[2]));
}

static bool parseScalarBgr(const std::string& s, cv::Scalar& out) {
    int b = 0, g = 0, r = 0;
    if (std::sscanf(s.c_str(), "%d,%d,%d", &b, &g, &r) == 3) {
        out = cv::Scalar((double)std::clamp(b, 0, 255), (double)std::clamp(g, 0, 255), (double)std::clamp(r, 0, 255));
        return true;
    }
    return false;
}

static void saveSettingsToDisk() {
    const std::filesystem::path path = getSettingsFilePath();
    try {
        std::filesystem::create_directories(path.parent_path());
    } catch (...) {
        // Best effort
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    f << "# BilliardsTrainer settings (auto-generated)\n";
    f << "# Format: key=value\n\n";

    // ---- UIControls ----
    f << "ui.selectedSource=" << uiControls.selectedSource << "\n";
    f << "ui.showOverlay=" << toStringBool(uiControls.showOverlay) << "\n";
    f << "ui.showFelt=" << toStringBool(uiControls.showFelt) << "\n";
    f << "ui.showRails=" << toStringBool(uiControls.showRails) << "\n";
    f << "ui.showDiamonds=" << toStringBool(uiControls.showDiamonds) << "\n";
    f << "ui.feltExpanded=" << toStringBool(uiControls.feltExpanded) << "\n";
    f << "ui.rectifyExpanded=" << toStringBool(uiControls.rectifyExpanded) << "\n";
    f << "ui.railsExpanded=" << toStringBool(uiControls.railsExpanded) << "\n";
    f << "ui.diamondsExpanded=" << toStringBool(uiControls.diamondsExpanded) << "\n";
    f << "ui.rectifyMarginScale=" << uiControls.rectifyMarginScale << "\n";
    f << "ui.rectifyPadPx=" << uiControls.rectifyPadPx << "\n";
    f << "ui.smoothingPercent=" << uiControls.smoothingPercent << "\n";

    // ---- Felt ----
    const auto& fp = uiControls.feltParams;
    f << "felt.hasPickedColor=" << toStringBool(fp.hasPickedColor) << "\n";
    f << "felt.pickedHSV=" << toStringVec3b(fp.pickedHSV) << "\n";
    f << "felt.pickedBGR=" << toStringVec3b(fp.pickedBGR) << "\n";
    f << "felt.colorSensitivity=" << fp.colorSensitivity << "\n";
    f << "felt.colorHMin=" << fp.colorHMin << "\n";
    f << "felt.colorHMax=" << fp.colorHMax << "\n";
    f << "felt.colorSMin=" << fp.colorSMin << "\n";
    f << "felt.colorSMax=" << fp.colorSMax << "\n";
    f << "felt.colorVMin=" << fp.colorVMin << "\n";
    f << "felt.colorVMax=" << fp.colorVMax << "\n";
    f << "felt.overlayColor=" << toStringScalarBgr(fp.color) << "\n";
    f << "felt.isFilled=" << toStringBool(fp.isFilled) << "\n";
    f << "felt.fillAlpha=" << fp.fillAlpha << "\n";
    f << "felt.outlineThicknessPx=" << fp.outlineThicknessPx << "\n";

    // ---- Rails ----
    const auto& rp = uiControls.railParams;
    f << "rail.hasPickedColor=" << toStringBool(rp.hasPickedColor) << "\n";
    f << "rail.pickedHSV=" << toStringVec3b(rp.pickedHSV) << "\n";
    f << "rail.pickedBGR=" << toStringVec3b(rp.pickedBGR) << "\n";
    f << "rail.colorSensitivity=" << rp.colorSensitivity << "\n";
    f << "rail.colorHMin=" << rp.colorHMin << "\n";
    f << "rail.colorHMax=" << rp.colorHMax << "\n";
    f << "rail.colorSMin=" << rp.colorSMin << "\n";
    f << "rail.colorSMax=" << rp.colorSMax << "\n";
    f << "rail.colorVMin=" << rp.colorVMin << "\n";
    f << "rail.colorVMax=" << rp.colorVMax << "\n";
    f << "rail.railBandWidth=" << rp.railBandWidth << "\n";
    f << "rail.railMaxWidth=" << rp.railMaxWidth << "\n";
    f << "rail.railMinArea=" << rp.railMinArea << "\n";
    f << "rail.railEdgeThreshold=" << rp.railEdgeThreshold << "\n";
    f << "rail.railEdgeThreshold2=" << rp.railEdgeThreshold2 << "\n";
    f << "rail.overlayColor=" << toStringScalarBgr(rp.color) << "\n";
    f << "rail.isFilled=" << toStringBool(rp.isFilled) << "\n";
    f << "rail.fillAlpha=" << rp.fillAlpha << "\n";
    f << "rail.outlineThicknessPx=" << rp.outlineThicknessPx << "\n";

    // ---- Diamonds ----
    const auto& dp = uiControls.diamondParams;
    f << "diamond.hasPickedColor=" << toStringBool(dp.hasPickedColor) << "\n";
    f << "diamond.pickedHSV=" << toStringVec3b(dp.pickedHSV) << "\n";
    f << "diamond.pickedBGR=" << toStringVec3b(dp.pickedBGR) << "\n";
    f << "diamond.colorSensitivity=" << dp.colorSensitivity << "\n";
    f << "diamond.colorHMin=" << dp.colorHMin << "\n";
    f << "diamond.colorHMax=" << dp.colorHMax << "\n";
    f << "diamond.colorSMin=" << dp.colorSMin << "\n";
    f << "diamond.colorSMax=" << dp.colorSMax << "\n";
    f << "diamond.colorVMin=" << dp.colorVMin << "\n";
    f << "diamond.colorVMax=" << dp.colorVMax << "\n";
    f << "diamond.minArea=" << dp.minArea << "\n";
    f << "diamond.maxArea=" << dp.maxArea << "\n";
    f << "diamond.edgeThreshold=" << dp.edgeThreshold << "\n";
    f << "diamond.edgeThreshold2=" << dp.edgeThreshold2 << "\n";
    f << "diamond.shortRailDiamonds=" << dp.shortRailDiamonds << "\n";
    f << "diamond.longRailDiamonds=" << dp.longRailDiamonds << "\n";
    f << "diamond.minDiamondSpacingPx=" << dp.minDiamondSpacingPx << "\n";
    f << "diamond.overlayColor=" << toStringScalarBgr(dp.color) << "\n";
    f << "diamond.isFilled=" << toStringBool(dp.isFilled) << "\n";
    f << "diamond.fillAlpha=" << dp.fillAlpha << "\n";
    f << "diamond.outlineThicknessPx=" << dp.outlineThicknessPx << "\n";
}

// Forward declarations for color sensitivity functions (defined later)
static void applyDiamondColorSensitivityToRangesFromPickedHSV();
static void applyFeltColorSensitivityToRangesFromPickedHSV();
static void applyRailColorSensitivityToRangesFromPickedHSV();

static void loadSettingsFromDisk() {
    const std::filesystem::path path = getSettingsFilePath();
    std::ifstream f(path, std::ios::binary);
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        line = trimCopy(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key = trimCopy(line.substr(0, eq));
        const std::string val = trimCopy(line.substr(eq + 1));

        // ---- UIControls ----
        if (key == "ui.selectedSource") { int v; if (parseInt(val, v)) uiControls.selectedSource = v; }
        else if (key == "ui.showOverlay") { bool b; if (parseBool(val, b)) uiControls.showOverlay = b; }
        else if (key == "ui.showFelt") { bool b; if (parseBool(val, b)) uiControls.showFelt = b; }
        else if (key == "ui.showRails") { bool b; if (parseBool(val, b)) uiControls.showRails = b; }
        else if (key == "ui.showDiamonds") { bool b; if (parseBool(val, b)) uiControls.showDiamonds = b; }
        else if (key == "ui.smoothingPercent") { int v; if (parseInt(val, v)) uiControls.smoothingPercent = std::clamp(v, 0, 100); }
        else if (key == "ui.feltExpanded") { bool b; if (parseBool(val, b)) uiControls.feltExpanded = b; }
        else if (key == "ui.rectifyExpanded") { bool b; if (parseBool(val, b)) uiControls.rectifyExpanded = b; }
        else if (key == "ui.railsExpanded") { bool b; if (parseBool(val, b)) uiControls.railsExpanded = b; }
        else if (key == "ui.diamondsExpanded") { bool b; if (parseBool(val, b)) uiControls.diamondsExpanded = b; }
        else if (key == "ui.rectifyMarginScale") { float v; if (std::sscanf(val.c_str(), "%f", &v) == 1) uiControls.rectifyMarginScale = std::clamp(v, 1.0f, 1.4f); }
        else if (key == "ui.rectifyPadPx") { int v; if (parseInt(val, v)) uiControls.rectifyPadPx = std::clamp(v, 0, 200); }

        // ---- Felt ----
        else if (key == "felt.hasPickedColor") { bool b; if (parseBool(val, b)) uiControls.feltParams.hasPickedColor = b; }
        else if (key == "felt.pickedHSV") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.feltParams.pickedHSV = vv; }
        else if (key == "felt.pickedBGR") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.feltParams.pickedBGR = vv; }
        else if (key == "felt.colorSensitivity") { int v; if (parseInt(val, v)) uiControls.feltParams.colorSensitivity = std::clamp(v, 0, 100); }
        else if (key == "felt.colorHMin") { int v; if (parseInt(val, v)) uiControls.feltParams.colorHMin = std::clamp(v, 0, 180); }
        else if (key == "felt.colorHMax") { int v; if (parseInt(val, v)) uiControls.feltParams.colorHMax = std::clamp(v, 0, 180); }
        else if (key == "felt.colorSMin") { int v; if (parseInt(val, v)) uiControls.feltParams.colorSMin = std::clamp(v, 0, 255); }
        else if (key == "felt.colorSMax") { int v; if (parseInt(val, v)) uiControls.feltParams.colorSMax = std::clamp(v, 0, 255); }
        else if (key == "felt.colorVMin") { int v; if (parseInt(val, v)) uiControls.feltParams.colorVMin = std::clamp(v, 0, 255); }
        else if (key == "felt.colorVMax") { int v; if (parseInt(val, v)) uiControls.feltParams.colorVMax = std::clamp(v, 0, 255); }
        else if (key == "felt.overlayColor") { cv::Scalar c; if (parseScalarBgr(val, c)) uiControls.feltParams.color = c; }
        else if (key == "felt.isFilled") { bool b; if (parseBool(val, b)) uiControls.feltParams.isFilled = b; }
        else if (key == "felt.fillAlpha") { int v; if (parseInt(val, v)) uiControls.feltParams.fillAlpha = std::clamp(v, 0, 255); }
        else if (key == "felt.outlineThicknessPx") { int v; if (parseInt(val, v)) uiControls.feltParams.outlineThicknessPx = std::max(1, v); }

        // ---- Rails ----
        else if (key == "rail.hasPickedColor") { bool b; if (parseBool(val, b)) uiControls.railParams.hasPickedColor = b; }
        else if (key == "rail.pickedHSV") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.railParams.pickedHSV = vv; }
        else if (key == "rail.pickedBGR") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.railParams.pickedBGR = vv; }
        else if (key == "rail.colorSensitivity") { int v; if (parseInt(val, v)) uiControls.railParams.colorSensitivity = std::clamp(v, 0, 100); }
        else if (key == "rail.colorHMin") { int v; if (parseInt(val, v)) uiControls.railParams.colorHMin = std::clamp(v, 0, 180); }
        else if (key == "rail.colorHMax") { int v; if (parseInt(val, v)) uiControls.railParams.colorHMax = std::clamp(v, 0, 180); }
        else if (key == "rail.colorSMin") { int v; if (parseInt(val, v)) uiControls.railParams.colorSMin = std::clamp(v, 0, 255); }
        else if (key == "rail.colorSMax") { int v; if (parseInt(val, v)) uiControls.railParams.colorSMax = std::clamp(v, 0, 255); }
        else if (key == "rail.colorVMin") { int v; if (parseInt(val, v)) uiControls.railParams.colorVMin = std::clamp(v, 0, 255); }
        else if (key == "rail.colorVMax") { int v; if (parseInt(val, v)) uiControls.railParams.colorVMax = std::clamp(v, 0, 255); }
        else if (key == "rail.railBandWidth") { int v; if (parseInt(val, v)) uiControls.railParams.railBandWidth = std::max(1, v); }
        else if (key == "rail.railMaxWidth") { int v; if (parseInt(val, v)) uiControls.railParams.railMaxWidth = std::max(1, v); }
        else if (key == "rail.railMinArea") { int v; if (parseInt(val, v)) uiControls.railParams.railMinArea = std::max(1, v); }
        else if (key == "rail.railEdgeThreshold") { int v; if (parseInt(val, v)) uiControls.railParams.railEdgeThreshold = std::max(1, v); }
        else if (key == "rail.railEdgeThreshold2") { int v; if (parseInt(val, v)) uiControls.railParams.railEdgeThreshold2 = std::max(1, v); }
        else if (key == "rail.overlayColor") { cv::Scalar c; if (parseScalarBgr(val, c)) uiControls.railParams.color = c; }
        else if (key == "rail.isFilled") { bool b; if (parseBool(val, b)) uiControls.railParams.isFilled = b; }
        else if (key == "rail.fillAlpha") { int v; if (parseInt(val, v)) uiControls.railParams.fillAlpha = std::clamp(v, 0, 255); }
        else if (key == "rail.outlineThicknessPx") { int v; if (parseInt(val, v)) uiControls.railParams.outlineThicknessPx = std::max(1, v); }

        // ---- Diamonds ----
        else if (key == "diamond.hasPickedColor") { bool b; if (parseBool(val, b)) uiControls.diamondParams.hasPickedColor = b; }
        else if (key == "diamond.pickedHSV") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.diamondParams.pickedHSV = vv; }
        else if (key == "diamond.pickedBGR") { cv::Vec3b vv; if (parseVec3b(val, vv)) uiControls.diamondParams.pickedBGR = vv; }
        else if (key == "diamond.colorSensitivity") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorSensitivity = std::clamp(v, 0, 100); }
        else if (key == "diamond.colorHMin") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorHMin = std::clamp(v, 0, 180); }
        else if (key == "diamond.colorHMax") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorHMax = std::clamp(v, 0, 180); }
        else if (key == "diamond.colorSMin") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorSMin = std::clamp(v, 0, 255); }
        else if (key == "diamond.colorSMax") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorSMax = std::clamp(v, 0, 255); }
        else if (key == "diamond.colorVMin") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorVMin = std::clamp(v, 0, 255); }
        else if (key == "diamond.colorVMax") { int v; if (parseInt(val, v)) uiControls.diamondParams.colorVMax = std::clamp(v, 0, 255); }
        else if (key == "diamond.minArea") { int v; if (parseInt(val, v)) uiControls.diamondParams.minArea = std::max(1, v); }
        else if (key == "diamond.maxArea") { int v; if (parseInt(val, v)) uiControls.diamondParams.maxArea = std::max(1, v); }
        else if (key == "diamond.edgeThreshold") { int v; if (parseInt(val, v)) uiControls.diamondParams.edgeThreshold = std::max(1, v); }
        else if (key == "diamond.edgeThreshold2") { int v; if (parseInt(val, v)) uiControls.diamondParams.edgeThreshold2 = std::max(1, v); }
        else if (key == "diamond.shortRailDiamonds") { int v; if (parseInt(val, v)) uiControls.diamondParams.shortRailDiamonds = std::max(0, v); }
        else if (key == "diamond.longRailDiamonds") { int v; if (parseInt(val, v)) uiControls.diamondParams.longRailDiamonds = std::max(0, v); }
        else if (key == "diamond.minDiamondSpacingPx") { float v; if (std::sscanf(val.c_str(), "%f", &v) == 1) uiControls.diamondParams.minDiamondSpacingPx = std::max(0.0f, v); }
        else if (key == "diamond.overlayColor") { cv::Scalar c; if (parseScalarBgr(val, c)) uiControls.diamondParams.color = c; }
        else if (key == "diamond.isFilled") { bool b; if (parseBool(val, b)) uiControls.diamondParams.isFilled = b; }
        else if (key == "diamond.fillAlpha") { int v; if (parseInt(val, v)) uiControls.diamondParams.fillAlpha = std::clamp(v, 0, 255); }
        else if (key == "diamond.outlineThicknessPx") { int v; if (parseInt(val, v)) uiControls.diamondParams.outlineThicknessPx = std::max(1, v); }
    }
    
    // After loading all settings, recompute HSV ranges if colors were picked
    // This ensures the ranges are consistent with the loaded sensitivity values
    // NOTE: Do NOT call updateColorPickerLabels() here as UI controls may not exist yet
    if (uiControls.diamondParams.hasPickedColor) {
        applyDiamondColorSensitivityToRangesFromPickedHSV();
    }
    if (uiControls.feltParams.hasPickedColor) {
        applyFeltColorSensitivityToRangesFromPickedHSV();
    }
    if (uiControls.railParams.hasPickedColor) {
        applyRailColorSensitivityToRangesFromPickedHSV();
    }
}

// Last rendered (overlaid) frame so menu-driven capture export can work.
static cv::Mat g_lastProcessedFrame;
static cv::Mat g_lastSourceFrame;  // Original unscaled source frame for color picking
static FeltDetectionResult g_lastFeltResult;  // Last felt detection result for export
static int g_lastFrameIndex = -1;  // Last processed frame index (for video)

// Latest frame analysis (accessible to UI and exporter)
FrameAnalysis latestAnalysis;


// Choose a deterministic capture directory.
// On Windows, we save next to the executable so "Export Captures" always goes somewhere predictable
// even if the process current working directory is unexpected (e.g. launched from Explorer / shortcuts).
static std::filesystem::path getCaptureDirectory() {
#ifdef _WIN32
    wchar_t modulePath[MAX_PATH] = {0};
    const DWORD n = GetModuleFileNameW(NULL, modulePath, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::filesystem::path exePath(modulePath);
        return exePath.parent_path() / "captures";
    }
#endif
    // Fallback: current working directory.
    return std::filesystem::current_path() / "captures";
}

// Forward declaration
extern HWND g_hwnd;

// Capture the entire application window as a screenshot
static cv::Mat captureWindowScreenshot(HWND hwnd) {
    if (!hwnd || !IsWindow(hwnd)) {
        return cv::Mat();
    }

    RECT windowRect;
    if (!GetWindowRect(hwnd, &windowRect)) {
        return cv::Mat();
    }

    const int width = windowRect.right - windowRect.left;
    const int height = windowRect.bottom - windowRect.top;
    if (width <= 0 || height <= 0) {
        return cv::Mat();
    }

    // Create device contexts
    HDC hdcScreen = GetDC(NULL);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, width, height);
    HGDIOBJ hOldBitmap = SelectObject(hdcMem, hBitmap);

    // Capture the full window including title bar and borders
    // Try PrintWindow first (better color handling), fallback to BitBlt
    BOOL printResult = PrintWindow(hwnd, hdcMem, 0);
    if (!printResult) {
        // Fallback: use BitBlt to copy directly from screen coordinates
        BitBlt(hdcMem, 0, 0, width, height, hdcScreen, windowRect.left, windowRect.top, SRCCOPY);
    }

    // Get bitmap info - use 24-bit RGB format
    BITMAPINFO bmpInfo = {};
    bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmpInfo.bmiHeader.biWidth = width;
    bmpInfo.bmiHeader.biHeight = -height; // Negative for top-down DIB
    bmpInfo.bmiHeader.biPlanes = 1;
    bmpInfo.bmiHeader.biBitCount = 24; // RGB (24-bit)
    bmpInfo.bmiHeader.biCompression = BI_RGB;

    // Allocate buffer for RGB data (DIB_RGB_COLORS returns RGB, not BGR)
    const int stride = ((width * 3 + 3) / 4) * 4; // Align to 4 bytes
    std::vector<unsigned char> rgbData(stride * height);

    // Get the bitmap bits - read raw bitmap format (BGR) without DIB_RGB_COLORS conversion
    // This should give us the bitmap in its native BGR format
    if (GetDIBits(hdcMem, hBitmap, 0, height, rgbData.data(), &bmpInfo, 0)) {
        // Create OpenCV Mat - copy data handling stride differences
        // Bitmap data is already in BGR format, so direct copy
        cv::Mat bgrMat(height, width, CV_8UC3);
        const int dstStride = bgrMat.step;
        for (int y = 0; y < height; ++y) {
            const unsigned char* src = rgbData.data() + y * stride;
            unsigned char* dst = bgrMat.data + y * dstStride;
            // Direct copy - bitmap is already BGR format
            std::memcpy(dst, src, width * 3);
        }

        // Cleanup
        SelectObject(hdcMem, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hdcMem);
        ReleaseDC(NULL, hdcScreen);

        return bgrMat;
    }

    // Cleanup on failure
    SelectObject(hdcMem, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(NULL, hdcScreen);

    return cv::Mat();
}

// Export the current overlay + diamond intermediate buffers to disk.
// Saved under a timestamped prefix: ./captures/capture-<ms-since-epoch>-*.png
static bool exportCapturesToDisk(const cv::Mat& processedImage, std::filesystem::path* outDir, std::string* outError) {
    if (outDir) *outDir = std::filesystem::path();
    if (outError) outError->clear();
    if (processedImage.empty()) {
        if (outError) *outError = "processedImage was empty (nothing to save).";
        return false;
    }

    const std::filesystem::path dir = getCaptureDirectory();
    if (outDir) *outDir = dir;
    try {
        std::filesystem::create_directories(dir);
    } catch (const std::exception& e) {
        if (outError) *outError = std::string("create_directories failed: ") + e.what();
        return false;
    } catch (...) {
        if (outError) *outError = "create_directories failed (unknown error).";
        return false;
    }

    const auto now = std::chrono::system_clock::now();
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const std::string stem = "capture-" + std::to_string(ts);

    struct WriteAttempt {
        std::filesystem::path path;
        bool ok = false;
        std::string error;
    };
    std::vector<WriteAttempt> attempts;
    attempts.reserve(8);

    auto writeImageChecked = [&](const std::filesystem::path& p, const cv::Mat& img) {
        WriteAttempt a;
        a.path = p;
        if (img.empty()) {
            a.ok = false;
            a.error = "image was empty";
            attempts.push_back(std::move(a));
            return;
        }
        try {
            const bool wrote = cv::imwrite(p.string(), img);
            if (!wrote) {
                a.ok = false;
                a.error = "cv::imwrite returned false";
            } else {
                // Verify the file actually exists and is non-empty.
                std::error_code ec;
                const bool exists = std::filesystem::exists(p, ec);
                const auto sz = exists ? std::filesystem::file_size(p, ec) : 0;
                a.ok = exists && sz > 0;
                if (!a.ok) {
                    a.error = "post-write verify failed (exists/size)";
                }
            }
        } catch (const std::exception& e) {
            a.ok = false;
            a.error = std::string("exception: ") + e.what();
        } catch (...) {
            a.ok = false;
            a.error = "exception: unknown";
        }
        attempts.push_back(std::move(a));
    };

    // Always attempt overlay first (live view with overlays).
    writeImageChecked(dir / (stem + "-overlay.png"), processedImage);

    // Export rectified image if available
    if (latestAnalysis.rect.ok && !latestAnalysis.rect.rectifiedBgr.empty()) {
        cv::Mat rectifiedWithOverlay = latestAnalysis.rect.rectifiedBgr.clone();
        
        // Apply felt mask overlay if enabled
        if (uiControls.showOverlay && uiControls.showFelt && !latestAnalysis.rect.rectifiedFeltMask.empty()) {
            cv::Mat mask = latestAnalysis.rect.rectifiedFeltMask;
            if (uiControls.feltParams.isFilled) {
                cv::Mat overlay = rectifiedWithOverlay.clone();
                // Create a 3-channel mask for blending
                std::vector<cv::Mat> maskChannels;
                cv::split(mask, maskChannels);
                cv::Mat mask3ch;
                cv::merge(std::vector<cv::Mat>{maskChannels[0], maskChannels[0], maskChannels[0]}, mask3ch);
                
                // Apply color overlay where mask is non-zero
                overlay.setTo(uiControls.feltParams.color, mask3ch);
                const double a = std::clamp(uiControls.feltParams.fillAlpha, 0, 255) / 255.0;
                cv::addWeighted(rectifiedWithOverlay, 1.0 - a, overlay, a, 0, rectifiedWithOverlay);
            }
        }
        
        writeImageChecked(dir / (stem + "-rect.png"), rectifiedWithOverlay);
    }


    // Export rectification debug images
    if (latestAnalysis.rect.ok) {
        // Export live_with_quads.png (live view with debug quads)
        cv::Mat liveWithQuads = processedImage.clone();
        // Draw felt corners quad (from felt_detection module) - tied to felt overlay toggle
        if (uiControls.showFelt) {
            drawFeltCornersQuad(liveWithQuads, g_lastFeltResult, uiControls.feltParams);
        }
        writeImageChecked(dir / (stem + "-live_with_quads.png"), liveWithQuads);
        
        // Export rect_with_guides.png (rectified view)
        cv::Mat rectWithGuides = latestAnalysis.rect.rectifiedBgr.clone();
        writeImageChecked(dir / (stem + "-rect_with_guides.png"), rectWithGuides);
    }

    // Export full window screenshot
    if (g_hwnd) {
        cv::Mat windowScreenshot = captureWindowScreenshot(g_hwnd);
        if (!windowScreenshot.empty()) {
            writeImageChecked(dir / (stem + "-window.png"), windowScreenshot);
        }
    }

    // Write felt detection telemetry JSON sidecar
    {
        const std::filesystem::path jsonPath = dir / (stem + ".json");
        try {
            std::ofstream jsonFile(jsonPath);
            if (jsonFile.is_open()) {
                jsonFile << "{\n";
                jsonFile << "  \"frame_index\": " << (g_lastFrameIndex >= 0 ? std::to_string(g_lastFrameIndex) : "null") << ",\n";
                jsonFile << "  \"image\": {\"w\":" << processedImage.cols << ",\"h\":" << processedImage.rows << "},\n";
                jsonFile << "  \"felt\": {\n";
                jsonFile << "    \"ok\": " << (g_lastFeltResult.ok ? "true" : "false") << ",\n";
                jsonFile << "    \"hasCorners\": " << (g_lastFeltResult.hasCorners ? "true" : "false") << ",\n";
                jsonFile << "    \"poly_size\": " << g_lastFeltResult.polySize << ",\n";
                jsonFile << "    \"found4Points\": " << (g_lastFeltResult.found4Points ? "true" : "false") << ",\n";
                jsonFile << "    \"areaRatio\": " << g_lastFeltResult.areaRatio << ",\n";
                jsonFile << "    \"envBox\": {\"x\":" << g_lastFeltResult.bbox.x << ",\"y\":" << g_lastFeltResult.bbox.y
                         << ",\"w\":" << g_lastFeltResult.bbox.width << ",\"h\":" << g_lastFeltResult.bbox.height << "},\n";
                jsonFile << "    \"hull_area\": " << g_lastFeltResult.envArea << ",\n";
                double contourArea = g_lastFeltResult.contour.empty() ? 0.0 : cv::contourArea(g_lastFeltResult.contour);
                jsonFile << "    \"contour_area\": " << contourArea << ",\n";
                
                // Corners
                jsonFile << "    \"corners_px\": ";
                if (g_lastFeltResult.hasCorners) {
                    jsonFile << "{\n";
                    jsonFile << "      \"TL\":{\"x\":" << g_lastFeltResult.corners[0].x << ",\"y\":" << g_lastFeltResult.corners[0].y << "},\n";
                    jsonFile << "      \"TR\":{\"x\":" << g_lastFeltResult.corners[1].x << ",\"y\":" << g_lastFeltResult.corners[1].y << "},\n";
                    jsonFile << "      \"BR\":{\"x\":" << g_lastFeltResult.corners[2].x << ",\"y\":" << g_lastFeltResult.corners[2].y << "},\n";
                    jsonFile << "      \"BL\":{\"x\":" << g_lastFeltResult.corners[3].x << ",\"y\":" << g_lastFeltResult.corners[3].y << "}\n";
                    jsonFile << "    },\n";
                } else {
                    jsonFile << "null,\n";
                }
                
                // HSV params
                jsonFile << "    \"hsv_params\": {\n";
                jsonFile << "      \"HMin\":" << uiControls.feltParams.colorHMin << ",\n";
                jsonFile << "      \"HMax\":" << uiControls.feltParams.colorHMax << ",\n";
                jsonFile << "      \"SMin\":" << uiControls.feltParams.colorSMin << ",\n";
                jsonFile << "      \"SMax\":" << uiControls.feltParams.colorSMax << ",\n";
                jsonFile << "      \"VMin\":" << uiControls.feltParams.colorVMin << ",\n";
                jsonFile << "      \"VMax\":" << uiControls.feltParams.colorVMax << "\n";
                jsonFile << "    }\n";
                jsonFile << "  },\n";
                
                // Rectification data
                jsonFile << "  \"rectification\": {\n";
                jsonFile << "    \"ok\": " << (latestAnalysis.rect.ok ? "true" : "false") << ",\n";
                jsonFile << "    \"rectifyMarginScale\": " << uiControls.rectifyMarginScale << ",\n";
                jsonFile << "    \"rectifyPadPx\": " << uiControls.rectifyPadPx << ",\n";
                if (latestAnalysis.rect.ok) {
                    jsonFile << "    \"dstSize\": {\"w\":" << latestAnalysis.rect.dstSize.width 
                             << ",\"h\":" << latestAnalysis.rect.dstSize.height << "},\n";
                    
                    // Source quad (original felt corners)
                    jsonFile << "    \"srcQuad\": [\n";
                    for (int i = 0; i < 4; ++i) {
                        jsonFile << "      {\"x\":" << latestAnalysis.rect.srcQuad[i].x 
                                 << ",\"y\":" << latestAnalysis.rect.srcQuad[i].y << "}";
                        if (i < 3) jsonFile << ",";
                        jsonFile << "\n";
                    }
                    jsonFile << "    ],\n";
                    
                    // Expanded source quad
                    jsonFile << "    \"expandedSrcQuad\": [\n";
                    for (int i = 0; i < 4; ++i) {
                        jsonFile << "      {\"x\":" << latestAnalysis.rect.expandedSrcQuad[i].x 
                                 << ",\"y\":" << latestAnalysis.rect.expandedSrcQuad[i].y << "}";
                        if (i < 3) jsonFile << ",";
                        jsonFile << "\n";
                    }
                    jsonFile << "    ],\n";
                    
                    // Destination quad
                    jsonFile << "    \"dstQuad\": [\n";
                    for (int i = 0; i < 4; ++i) {
                        jsonFile << "      {\"x\":" << latestAnalysis.rect.dstQuad[i].x 
                                 << ",\"y\":" << latestAnalysis.rect.dstQuad[i].y << "}";
                        if (i < 3) jsonFile << ",";
                        jsonFile << "\n";
                    }
                    jsonFile << "    ],\n";
                    
                    // Write homography matrix H
                    if (!latestAnalysis.rect.H.empty() && latestAnalysis.rect.H.rows == 3 && latestAnalysis.rect.H.cols == 3) {
                        jsonFile << "    \"H\": [\n";
                        for (int i = 0; i < 3; ++i) {
                            jsonFile << "      [";
                            for (int j = 0; j < 3; ++j) {
                                jsonFile << latestAnalysis.rect.H.at<double>(i, j);
                                if (j < 2) jsonFile << ",";
                            }
                            jsonFile << "]";
                            if (i < 2) jsonFile << ",";
                            jsonFile << "\n";
                        }
                        jsonFile << "    ],\n";
                    } else {
                        jsonFile << "    \"H\": null,\n";
                    }
                    
                    // Write inverse homography matrix Hinv
                    if (!latestAnalysis.rect.Hinv.empty() && latestAnalysis.rect.Hinv.rows == 3 && latestAnalysis.rect.Hinv.cols == 3) {
                        jsonFile << "    \"Hinv\": [\n";
                        for (int i = 0; i < 3; ++i) {
                            jsonFile << "      [";
                            for (int j = 0; j < 3; ++j) {
                                jsonFile << latestAnalysis.rect.Hinv.at<double>(i, j);
                                if (j < 2) jsonFile << ",";
                            }
                            jsonFile << "]";
                            if (i < 2) jsonFile << ",";
                            jsonFile << "\n";
                        }
                        jsonFile << "    ]\n";
                    } else {
                        jsonFile << "    \"Hinv\": null\n";
                    }
                } else {
                    jsonFile << "    \"dstSize\": null,\n";
                    jsonFile << "    \"H\": null,\n";
                    jsonFile << "    \"Hinv\": null\n";
                }
                jsonFile << "  }\n";
                jsonFile << "}\n";
                jsonFile.close();
            }
        } catch (...) {
            // ignore JSON write errors
        }
    }
    
    // Always write a manifest so we can diagnose "it said exported but folder is empty".
    // This also acts as a sanity check that we can write *something* to the directory.
    {
        const std::filesystem::path manifestPath = dir / (stem + "-manifest.txt");
        try {
            FILE* f = nullptr;
#ifdef _WIN32
            _wfopen_s(&f, manifestPath.wstring().c_str(), L"wb");
#else
            f = std::fopen(manifestPath.string().c_str(), "wb");
#endif
            if (f) {
                std::string header = "BilliardsTrainer capture manifest\n";
                header += "timestampMs=" + std::to_string(ts) + "\n";
                header += "dir=" + dir.string() + "\n";
                for (const auto& a : attempts) {
                    header += (a.ok ? "OK " : "FAIL ");
                    header += a.path.filename().string();
                    if (!a.ok && !a.error.empty()) header += " (" + a.error + ")";
                    header += "\n";
                }
                (void)std::fwrite(header.data(), 1, header.size(), f);
                std::fclose(f);
            }
        } catch (...) {
            // ignore
        }
    }

    int okCount = 0;
    for (const auto& a : attempts) if (a.ok) okCount++;

    if (okCount <= 0) {
        if (outError) {
            std::string msg = "No files were verified on disk.\n";
            for (const auto& a : attempts) {
                msg += "- " + a.path.filename().string() + ": " + (a.ok ? "OK" : "FAIL");
                if (!a.ok && !a.error.empty()) msg += " (" + a.error + ")";
                msg += "\n";
            }
            *outError = msg;
        }
        return false;
    }

    return true;
}
std::vector<int> enumerateCameras();
void createNativeMenu(HWND hwnd, const std::vector<int>& cameras);
void handleMenuCommand(int menuId);
void updateOverlayMenu();
void updateLayout(HWND hwnd);  // Update container layout (image area and sidebar)
void createSidebarControls(HWND hwnd);
void destroySidebarControls();
void updateSidebarControls();
void handleTrackbarChange(int trackbarId, int value);
void handleSidebarButton(int buttonId);
LRESULT CALLBACK SidebarContainerProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
HWND g_hwnd = NULL;
WNDPROC g_oldWndProc = NULL;
HWND g_sidebarPanel = NULL;      // The sidebar panel window (child of main window, always on top)
HWND g_imageContainer = NULL;     // Container for the OpenCV image display area
int g_sidebarScrollPos = 0;      // Current scroll position
int g_sidebarContentHeight = 0;  // Total height of sidebar content
std::vector<int> g_availableCameras;

#ifdef _WIN32
// ================================================================================================
// Win32-hosted UI (side-by-side containers)
// - The OS-owned "main window" hosts two child windows:
//   - left: ImageView (renders pixels)
//   - right: SidebarPanel (native controls)
// - OpenCV does NOT own the main window. OpenCV only produces cv::Mat frames.
// ================================================================================================

// Color picker button ID (defined early for use in mouse handlers)
#define IDC_DIAMOND_COLOR_PICKER 30251
// Felt color picker button ID
#define IDC_FELT_COLOR_PICKER 30252
// Rail color picker button ID
#define IDC_RAIL_COLOR_PICKER 30253

// Diamond color picker info labels (defined early for use in update function)
#define IDC_DIAMOND_COLOR_PICKER_BGR 10050
#define IDC_DIAMOND_COLOR_PICKER_HSV 10051
#define IDC_DIAMOND_COLOR_PICKER_RANGE 10052
#define IDC_DIAMOND_COLOR_PICKER_SWATCH 10055

// Felt color picker info labels (defined early for use in update function)
#define IDC_FELT_COLOR_PICKER_BGR 10060
#define IDC_FELT_COLOR_PICKER_HSV 10061
#define IDC_FELT_COLOR_PICKER_RANGE 10062
#define IDC_FELT_COLOR_PICKER_SWATCH 10065

// Rail color picker info labels (defined early for use in update function)
#define IDC_RAIL_COLOR_PICKER_BGR 10070
#define IDC_RAIL_COLOR_PICKER_HSV 10071
#define IDC_RAIL_COLOR_PICKER_RANGE 10072
#define IDC_RAIL_COLOR_PICKER_SWATCH 10075

#define IDC_COMBO_BASE 40000

static HWND g_imageViewHwnd = NULL;
static HWND g_rectifiedViewHwnd = NULL;  // Second viewport for top-down rectified view
static HWND g_zoomWindowHwnd = NULL;  // Zoom window for color picker
static HBRUSH g_imageBgBrush = NULL;
static HBRUSH g_sidebarBgBrush = NULL;
// Color picker swatch brush:
// - The swatch is a STATIC control. To paint it correctly and reliably, we provide a brush via
//   WM_CTLCOLORSTATIC from the SidebarPanelProc.
// - We own this brush and must delete it on replacement / shutdown to avoid leaking GDI objects.
static HBRUSH g_diamondColorPickerSwatchBrush = NULL;
static HBRUSH g_feltColorPickerSwatchBrush = NULL;
static HBRUSH g_railColorPickerSwatchBrush = NULL;
static HFONT g_sidebarFont = NULL;
static HFONT g_sidebarFontBold = NULL;

// Which target (if any) the color picker is currently sampling for.
enum class ColorPickerTarget {
    None,
    Diamond,
    Felt,
    Rail
};
static ColorPickerTarget g_colorPickerTarget = ColorPickerTarget::None;
static int g_scaledImageX = 0, g_scaledImageY = 0, g_scaledImageW = 0, g_scaledImageH = 0;  // Scaled image position/size for coordinate conversion

static void applyFont(HWND hwnd, bool isBold = false) {
    HFONT font = isBold ? g_sidebarFontBold : g_sidebarFont;
    if (font && hwnd && IsWindow(hwnd)) {
        SendMessage(hwnd, WM_SETFONT, (WPARAM)font, TRUE);
    }
}

static COLORREF colorRefFromBgrScalar(const cv::Scalar& bgr) {
    const int b = (int)std::clamp(bgr[0], 0.0, 255.0);
    const int g = (int)std::clamp(bgr[1], 0.0, 255.0);
    const int r = (int)std::clamp(bgr[2], 0.0, 255.0);
    return RGB(r, g, b);
}

static cv::Scalar bgrScalarFromColorRef(COLORREF c) {
    return cv::Scalar((double)GetBValue(c), (double)GetGValue(c), (double)GetRValue(c));
}

static bool chooseColor(HWND owner, const cv::Scalar& initialBgr, cv::Scalar& outBgr) {
    COLORREF custom[16]{};
    CHOOSECOLORW cc{};
    cc.lStructSize = sizeof(cc);
    cc.hwndOwner = owner;
    cc.lpCustColors = custom;
    cc.rgbResult = colorRefFromBgrScalar(initialBgr);
    cc.Flags = CC_FULLOPEN | CC_RGBINIT;
    if (ChooseColorW(&cc)) {
        outBgr = bgrScalarFromColorRef(cc.rgbResult);
        return true;
    }
    return false;
}

// Backing store for the ImageView (32bpp BGRA, top-down DIB)
struct ImageDibBuffer {
    BITMAPINFO bmi{};
    int width = 0;
    int height = 0;
    std::vector<unsigned char> bgra; // width * height * 4
};

static ImageDibBuffer g_imageDib;
static ImageDibBuffer g_rectifiedDib;  // DIB buffer for rectified view

// App data sources
static cv::Mat g_testImage;
static cv::VideoCapture g_camera;
static std::string g_testVideoPath = "testVideo.mp4";

// ------------------------------------------------------------------------------------------------
// Camera capture runs on a background thread.
//
// Rationale:
// - On some systems/backends, opening a camera and/or reading frames can block for noticeable time.
// - If we do that work on the UI thread, the app feels like it “doesn’t switch” (especially when
//   the user immediately switches back to Test Image).
// - The UI thread should always be able to render the test image immediately.
//
// Design:
// - `g_requestedCaptureSource`:
//     - -1 means "Test Image"
//     - -2 means "Test Video" (looping file)
//     - >= 0 means "open that camera index"
// - The capture thread updates `g_latestCaptureFrame` whenever it successfully reads a frame.
// - The UI thread renders:
//   - test image directly when selected (never waits on capture thread)
//   - else: latest capture frame (camera/video) if available, otherwise test image as fallback.
// ------------------------------------------------------------------------------------------------
// Start by requesting whatever the UI defaults to (currently Test Video).
static std::atomic<int> g_requestedCaptureSource{uiControls.selectedSource};
static std::atomic<bool> g_captureThreadRunning{false};
static std::thread g_captureThread;
static std::mutex g_latestFrameMutex;
static cv::Mat g_latestCaptureFrame;

static void stopCaptureThread() {
    g_captureThreadRunning.store(false, std::memory_order_relaxed);
    if (g_captureThread.joinable()) {
        g_captureThread.join();
    }
}

static void startCaptureThread() {
    // Ensure only one capture thread exists.
    stopCaptureThread();
    g_captureThreadRunning.store(true, std::memory_order_relaxed);

    g_captureThread = std::thread([]() {
        cv::VideoCapture cap;
        int currentSource = -999; // sentinel; forces first open attempt

        auto openTestVideo = [&](cv::VideoCapture& ioCap) -> bool {
            if (ioCap.isOpened()) ioCap.release();
            ioCap.open(g_testVideoPath);
            if (!ioCap.isOpened()) ioCap.open("testVideo.mp4");
            if (!ioCap.isOpened()) ioCap.open("../../testVideo.mp4");
            return ioCap.isOpened();
        };

        auto rewindTestVideoOrReopen = [&](cv::VideoCapture& ioCap) -> bool {
            if (!ioCap.isOpened()) return openTestVideo(ioCap);

            // Strategy 1: frame seek (works on many demuxers/codecs)
            const bool seekFramesOk = ioCap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (seekFramesOk) return true;

            // Strategy 2: ratio seek (sometimes works when frame seek is ignored)
            const bool seekRatioOk = ioCap.set(cv::CAP_PROP_POS_AVI_RATIO, 0.0);
            if (seekRatioOk) return true;

            // Strategy 3: reopen the file (most reliable fallback)
            return openTestVideo(ioCap);
        };

        while (g_captureThreadRunning.load(std::memory_order_relaxed)) {
            const int requested = g_requestedCaptureSource.load(std::memory_order_relaxed);

            // Switch camera / release camera as needed.
            if (requested != currentSource) {
                if (cap.isOpened()) {
                    cap.release();
                }

                currentSource = requested;

                if (currentSource >= 0) {
                    // Prefer a backend that tends to behave well on Windows.
                    // If this fails, OpenCV will fall back internally on some installs;
                    // but using an explicit backend often reduces open latency/hangs.
                    cap.open(currentSource, cv::CAP_DSHOW);
                    if (cap.isOpened()) {
                        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                    }
                } else if (currentSource == kSourceTestVideo) {
                    // Test video: open file source (looped).
                    //
                    // We keep this logic inside the capture thread so switching sources never blocks the UI thread.
                    // The path is resolved in `runWin32HostedApp()` (and in the non-hosted path we also try a
                    // relative fallback).
                    openTestVideo(cap);
                }
            }

            // If test image (or anything unopened), just idle.
            if (currentSource == kSourceTestImage || !cap.isOpened()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            cv::Mat frame;
            // NOTE: Some backends block in read(). If that happens, UI still stays responsive because
            // this thread is the only one affected. UI can always switch back to Test Image.
            if (currentSource == kSourceTestVideo) {
                // Proactive EOF handling: if we're at/near the end, rewind before read.
                const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
                const double posFrames = cap.get(cv::CAP_PROP_POS_FRAMES);
                if (frameCount > 1.0 && posFrames >= frameCount - 1.0) {
                    rewindTestVideoOrReopen(cap);
                }
            }
            if (!cap.read(frame) || frame.empty()) {
                if (currentSource == kSourceTestVideo) {
                    // EOF / read hiccup: robust loop.
                    // Some codecs/demuxers don't reliably honor frame seeking after long playback;
                    // reopening is the fallback to prevent the UI from "freezing" on the last frame.
                    if (!rewindTestVideoOrReopen(cap)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                    // Try to read again after rewind/reopen.
                    if (!cap.read(frame) || frame.empty()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                } else {
                    // Backoff a bit on camera read failure.
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    continue;
                }
            }

            {
                std::lock_guard<std::mutex> lock(g_latestFrameMutex);
                g_latestCaptureFrame = frame;
            }
            
            // Track frame index for export (after successful read)
            if (currentSource == kSourceTestVideo && cap.isOpened()) {
                g_lastFrameIndex = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1; // -1 because read advances
            } else {
                g_lastFrameIndex = -1; // Camera or image input, no frame index
            }

            // For video files, respect the file's FPS to avoid burning CPU / skipping too fast.
            if (currentSource == kSourceTestVideo) {
                const double fps = cap.get(cv::CAP_PROP_FPS);
                const int delayMs = (fps > 1.0 && fps < 240.0) ? (int)std::lround(1000.0 / fps) : 33;
                std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, delayMs)));
            }
        }

        if (cap.isOpened()) {
            cap.release();
        }
    });
}

static const wchar_t* kMainWindowClass = L"BilliardsTrainerMainWindow";
static const wchar_t* kImageViewClass = L"BilliardsTrainerImageView";
static const wchar_t* kSidebarPanelClass = L"BilliardsTrainerSidebarPanel";

static void ensureSidebarAndImageChildren(HWND mainHwnd);
static void layoutChildren(HWND mainHwnd);
static void updateColorPickerLabels();
static void applyDiamondColorSensitivityToRangesFromPickedHSV();
static void applyFeltColorSensitivityToRangesFromPickedHSV();
static void applyRailColorSensitivityToRangesFromPickedHSV();
static void updateImageDibFromBgr(const cv::Mat& bgr);

// Update the global swatch brush used by WM_CTLCOLORSTATIC.
static void setSwatchBrush(HBRUSH& ioBrush, COLORREF swatchColor) {
    if (ioBrush) {
        DeleteObject(ioBrush);
        ioBrush = NULL;
    }
    ioBrush = CreateSolidBrush(swatchColor);
}

// SidebarPanel WndProc:
// Child controls (buttons/trackbars) send WM_COMMAND / WM_HSCROLL to their *parent* (the panel),
// not to the main window. We forward these messages to the main window so clicks/sliders work.
static LRESULT CALLBACK SidebarPanelProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_COMMAND:
        case WM_HSCROLL: {
            HWND parent = GetParent(hwnd);
            if (parent && IsWindow(parent)) {
                return SendMessage(parent, msg, wParam, lParam);
            }
            break;
        }
        case WM_CTLCOLORSTATIC: {
            // Provide a custom brush for the color picker swatch so the fill color is stable and
            // does not rely on undefined SetClassLongPtr(GCLP_HBRBACKGROUND) behavior.
            HDC hdc = (HDC)wParam;
            HWND child = (HWND)lParam;
            const int childId = child ? GetDlgCtrlID(child) : 0;
            if (child && childId == IDC_DIAMOND_COLOR_PICKER_SWATCH && g_diamondColorPickerSwatchBrush) {
                // It's a solid color block: no text, so just set a matching background.
                // (Text color doesn't matter because the swatch has an empty caption.)
                SetBkMode(hdc, OPAQUE);
                return (INT_PTR)g_diamondColorPickerSwatchBrush;
            }
            if (child && childId == IDC_FELT_COLOR_PICKER_SWATCH && g_feltColorPickerSwatchBrush) {
                SetBkMode(hdc, OPAQUE);
                return (INT_PTR)g_feltColorPickerSwatchBrush;
            }
            if (child && childId == IDC_RAIL_COLOR_PICKER_SWATCH && g_railColorPickerSwatchBrush) {
                SetBkMode(hdc, OPAQUE);
                return (INT_PTR)g_railColorPickerSwatchBrush;
            }
            break;
        }
        case WM_VSCROLL: {
            // Container-level scrolling (WS_VSCROLL on the sidebar panel).
            SCROLLINFO si{};
            si.cbSize = sizeof(si);
            si.fMask = SIF_POS | SIF_RANGE | SIF_PAGE;
            GetScrollInfo(hwnd, SB_VERT, &si);

            const int oldPos = si.nPos;
            int newPos = oldPos;

            switch (LOWORD(wParam)) {
                case SB_LINEUP: newPos -= 20; break;
                case SB_LINEDOWN: newPos += 20; break;
                case SB_PAGEUP: newPos -= (int)si.nPage; break;
                case SB_PAGEDOWN: newPos += (int)si.nPage; break;
                case SB_THUMBTRACK:
                case SB_THUMBPOSITION: newPos = HIWORD(wParam); break;
                case SB_TOP: newPos = si.nMin; break;
                case SB_BOTTOM: newPos = si.nMax; break;
            }

            const int maxPos = std::max(si.nMin, (int)(si.nMax - (int)si.nPage + 1));
            newPos = std::max(si.nMin, std::min(newPos, maxPos));

            if (newPos != oldPos) {
                g_sidebarScrollPos = newPos;
                si.fMask = SIF_POS;
                si.nPos = newPos;
                SetScrollInfo(hwnd, SB_VERT, &si, TRUE);
                // Smooth scrolling: move existing child windows instead of destroying/recreating everything.
                // This avoids flicker and feels like a normal native scroll container.
                const int dy = oldPos - newPos;
                ScrollWindowEx(hwnd, 0, dy, NULL, NULL, NULL, NULL, SW_INVALIDATE | SW_SCROLLCHILDREN);
                UpdateWindow(hwnd);
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            // Mouse wheel scroll for sidebar content
            const int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            SCROLLINFO si{};
            si.cbSize = sizeof(si);
            si.fMask = SIF_POS | SIF_RANGE | SIF_PAGE;
            GetScrollInfo(hwnd, SB_VERT, &si);

            const int oldPos = si.nPos;
            int newPos = oldPos - (delta / WHEEL_DELTA) * 40;
            const int maxPos = std::max(si.nMin, (int)(si.nMax - (int)si.nPage + 1));
            newPos = std::max(si.nMin, std::min(newPos, maxPos));

            if (newPos != oldPos) {
                g_sidebarScrollPos = newPos;
                si.fMask = SIF_POS;
                si.nPos = newPos;
                SetScrollInfo(hwnd, SB_VERT, &si, TRUE);
                const int dy = oldPos - newPos;
                ScrollWindowEx(hwnd, 0, dy, NULL, NULL, NULL, NULL, SW_INVALIDATE | SW_SCROLLCHILDREN);
                UpdateWindow(hwnd);
                return 0;
            }
            break;
        }
        case WM_ERASEBKGND: {
            // Reduce flicker: we'll paint background as part of WM_PAINT.
            return 1;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);
            RECT rc{};
            GetClientRect(hwnd, &rc);
            if (!g_sidebarBgBrush) {
                g_sidebarBgBrush = CreateSolidBrush(RGB(240, 240, 240));
            }
            FillRect(hdc, &rc, g_sidebarBgBrush);
            EndPaint(hwnd, &ps);
            return 0;
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Native Windows magnifier window
static const wchar_t* kMagnifierClass = L"BilliardsTrainerMagnifier";
static HWND g_magnifierHwnd = NULL;
static HDC g_magnifierDC = NULL;
static HBITMAP g_magnifierBitmap = NULL;
static int g_magnifierSize = 200;

// Magnifier window procedure
static LRESULT CALLBACK MagnifierProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            RECT rc;
            GetClientRect(hwnd, &rc);
            
            if (g_magnifierBitmap && g_magnifierDC) {
                HDC memDC = CreateCompatibleDC(hdc);
                HGDIOBJ oldBmp = SelectObject(memDC, g_magnifierBitmap);
                BitBlt(hdc, 0, 0, rc.right, rc.bottom, memDC, 0, 0, SRCCOPY);
                SelectObject(memDC, oldBmp);
                DeleteDC(memDC);
            } else {
                FillRect(hdc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));
            }
            
            // Draw border
            HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 255, 255));
            HGDIOBJ oldPen = SelectObject(hdc, hPen);
            HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
            Rectangle(hdc, 1, 1, rc.right - 1, rc.bottom - 1);
            SelectObject(hdc, oldBrush);
            SelectObject(hdc, oldPen);
            DeleteObject(hPen);
            
            // Draw crosshairs
            int centerX = rc.right / 2;
            int centerY = rc.bottom / 2;
            HPEN crossPen = CreatePen(PS_SOLID, 1, RGB(0, 255, 255));
            SelectObject(hdc, crossPen);
            MoveToEx(hdc, centerX, 0, NULL);
            LineTo(hdc, centerX, rc.bottom);
            MoveToEx(hdc, 0, centerY, NULL);
            LineTo(hdc, rc.right, centerY);
            SelectObject(hdc, oldPen);
            DeleteObject(crossPen);
            
            // Draw center dot
            HBRUSH dotBrush = CreateSolidBrush(RGB(255, 0, 0));
            HGDIOBJ oldDotBrush = SelectObject(hdc, dotBrush);
            Ellipse(hdc, centerX - 3, centerY - 3, centerX + 3, centerY + 3);
            SelectObject(hdc, oldDotBrush);
            DeleteObject(dotBrush);
            
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_ERASEBKGND:
            return 1;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Create magnifier window
static void createMagnifierWindow() {
    if (g_magnifierHwnd) return;
    
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MagnifierProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = kMagnifierClass;
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.hCursor = LoadCursor(NULL, IDC_CROSS);
    RegisterClassW(&wc);
    
    g_magnifierHwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED,
        kMagnifierClass,
        L"Color Picker Magnifier",
        WS_POPUP | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        g_magnifierSize, g_magnifierSize,
        NULL, NULL,
        GetModuleHandleW(NULL),
        NULL
    );
    
    if (g_magnifierHwnd) {
        SetLayeredWindowAttributes(g_magnifierHwnd, 0, 255, LWA_ALPHA);
        ShowWindow(g_magnifierHwnd, SW_HIDE);
    }
}

// Update magnifier window with zoomed image
static void updateMagnifierWindow(int imgX, int imgY, int screenX, int screenY) {
    if (g_colorPickerTarget == ColorPickerTarget::None || g_lastSourceFrame.empty()) {
        if (g_magnifierHwnd) ShowWindow(g_magnifierHwnd, SW_HIDE);
        return;
    }
    
    if (!g_magnifierHwnd) {
        createMagnifierWindow();
    }
    
    if (!g_magnifierHwnd) return;
    
    const int sampleSize = 30;
    int halfSample = sampleSize / 2;
    int x1 = std::max(0, imgX - halfSample);
    int y1 = std::max(0, imgY - halfSample);
    int x2 = std::min(g_lastSourceFrame.cols, imgX + halfSample);
    int y2 = std::min(g_lastSourceFrame.rows, imgY + halfSample);
    
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    if (roi.width <= 0 || roi.height <= 0) return;
    
    cv::Mat zoomRegion = g_lastSourceFrame(roi);
    cv::Mat zoomed;
    cv::resize(zoomRegion, zoomed, cv::Size(g_magnifierSize, g_magnifierSize), 0, 0, cv::INTER_CUBIC);
    
    // Convert to BGRA for Windows
    cv::Mat bgra;
    cv::cvtColor(zoomed, bgra, cv::COLOR_BGR2BGRA);
    
    // Create or update bitmap
    if (!g_magnifierDC) {
        HDC screenDC = GetDC(NULL);
        g_magnifierDC = CreateCompatibleDC(screenDC);
        ReleaseDC(NULL, screenDC);
    }
    
    if (g_magnifierBitmap) {
        DeleteObject(g_magnifierBitmap);
    }
    
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = g_magnifierSize;
    bmi.bmiHeader.biHeight = -g_magnifierSize; // Top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    
    void* bits = NULL;
    g_magnifierBitmap = CreateDIBSection(g_magnifierDC, &bmi, DIB_RGB_COLORS, &bits, NULL, 0);
    if (g_magnifierBitmap && bits) {
        memcpy(bits, bgra.data, g_magnifierSize * g_magnifierSize * 4);
    }
    
    // Position window
    const int offsetX = 40;
    const int offsetY = 40;
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    
    int posX = screenX + offsetX;
    int posY = screenY + offsetY;
    if (posX + g_magnifierSize > screenW) posX = screenX - g_magnifierSize - offsetX;
    if (posY + g_magnifierSize > screenH) posY = screenY - g_magnifierSize - offsetY;
    if (posX < 0) posX = 10;
    if (posY < 0) posY = 10;
    
    SetWindowPos(g_magnifierHwnd, HWND_TOPMOST, posX, posY, 0, 0, SWP_NOSIZE | SWP_SHOWWINDOW);
    InvalidateRect(g_magnifierHwnd, NULL, TRUE);
    UpdateWindow(g_magnifierHwnd);
}

// OpenCV mouse callback for color picker (used when OpenCV window is active)
static void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (g_colorPickerTarget == ColorPickerTarget::None || g_lastSourceFrame.empty()) return;
    
    // Convert display coordinates to source image coordinates
    // The displayed image is scaled, so we need to account for that
    int imgX, imgY;
    if (g_scaledImageW > 0 && g_scaledImageH > 0) {
        // Scale coordinates back to source image
        imgX = (int)((double)x / g_scaledImageW * g_lastSourceFrame.cols);
        imgY = (int)((double)y / g_scaledImageH * g_lastSourceFrame.rows);
        imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
        imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
    } else {
        imgX = x;
        imgY = y;
    }
    
    // Get screen coordinates for positioning zoom window
    POINT pt;
    GetCursorPos(&pt);  // Get current cursor position in screen coordinates
    
    if (event == cv::EVENT_MOUSEMOVE) {
        updateMagnifierWindow(imgX, imgY, pt.x, pt.y);
    } else if (event == cv::EVENT_LBUTTONDOWN) {
        // Sample color at click position
        cv::Vec3b bgr = g_lastSourceFrame.at<cv::Vec3b>(imgY, imgX);
        
        // Convert BGR to HSV
        cv::Mat bgrMat(1, 1, CV_8UC3);
        bgrMat.at<cv::Vec3b>(0, 0) = bgr;
        cv::Mat hsvMat;
        cv::cvtColor(bgrMat, hsvMat, cv::COLOR_BGR2HSV);
        cv::Vec3b hsv = hsvMat.at<cv::Vec3b>(0, 0);
        
        // Store the picked color into the active target.
        if (g_colorPickerTarget == ColorPickerTarget::Felt) {
            uiControls.feltParams.pickedBGR = bgr;
            uiControls.feltParams.pickedHSV = hsv;
            uiControls.feltParams.hasPickedColor = true;
            applyFeltColorSensitivityToRangesFromPickedHSV();
        } else if (g_colorPickerTarget == ColorPickerTarget::Rail) {
            uiControls.railParams.pickedBGR = bgr;
            uiControls.railParams.pickedHSV = hsv;
            uiControls.railParams.hasPickedColor = true;
            applyRailColorSensitivityToRangesFromPickedHSV();
        }

        // Hide magnifier and deactivate picker FIRST, then refresh UI.
        // `updateColorPickerLabels()` uses `g_colorPickerTarget` to decide the button titles.
        if (g_magnifierHwnd) {
            ShowWindow(g_magnifierHwnd, SW_HIDE);
        }
        g_colorPickerTarget = ColorPickerTarget::None;

        // Update UI labels (also updates button text)
        updateColorPickerLabels();
        
        // Save settings immediately when color is picked
        saveSettingsToDisk();
    }
}

// MOVE?
// Helper function to calculate aspect-ratio-preserving scaled image bounds
static void calculateScaledImageBounds(int imageW, int imageH, int viewportW, int viewportH,
                                        int& outScaledW, int& outScaledH, int& outOffsetX, int& outOffsetY) {
    if (imageW <= 0 || imageH <= 0 || viewportW <= 0 || viewportH <= 0) {
        outScaledW = 0;
        outScaledH = 0;
        outOffsetX = 0;
        outOffsetY = 0;
        return;
    }

    const double imageAspect = static_cast<double>(imageW) / static_cast<double>(imageH);
    const double viewportAspect = static_cast<double>(viewportW) / static_cast<double>(viewportH);
    
    if (imageAspect > viewportAspect) {
        // Image is wider - fit to width, letterbox top/bottom
        outScaledW = viewportW;
        outScaledH = static_cast<int>(viewportW / imageAspect);
        outOffsetX = 0;
        outOffsetY = (viewportH - outScaledH) / 2;
    } else {
        // Image is taller - fit to height, pillarbox left/right
        outScaledH = viewportH;
        outScaledW = static_cast<int>(viewportH * imageAspect);
        outOffsetX = (viewportW - outScaledW) / 2;
        outOffsetY = 0;
    }
}


// ImageView WndProc: draws the latest DIB scaled to fit.
static LRESULT CALLBACK ImageViewProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_ERASEBKGND: {
            // Prevent background erase flicker; we fully paint in WM_PAINT.
            return 1;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT rc{};
            GetClientRect(hwnd, &rc);
            const int dstW = std::max(0L, rc.right - rc.left);
            const int dstH = std::max(0L, rc.bottom - rc.top);

            // Double-buffer to reduce flicker.
            HDC memDC = CreateCompatibleDC(hdc);
            HBITMAP memBmp = CreateCompatibleBitmap(hdc, dstW, dstH);
            HGDIOBJ oldBmp = SelectObject(memDC, memBmp);

            // Fill background
            if (!g_imageBgBrush) g_imageBgBrush = CreateSolidBrush(RGB(30, 30, 30));
            FillRect(memDC, &rc, g_imageBgBrush);

            // Determine which DIB buffer to use based on which window is being painted
            const bool isRectifiedView = (hwnd == g_rectifiedViewHwnd);
            const ImageDibBuffer& dib = isRectifiedView ? g_rectifiedDib : g_imageDib;

            if (dib.width > 0 && dib.height > 0 && !dib.bgra.empty()) {
                SetStretchBltMode(memDC, HALFTONE);

                // Calculate aspect-ratio-preserving dimensions
                int scaledW, scaledH, offsetX, offsetY;
                calculateScaledImageBounds(dib.width, dib.height, dstW, dstH, scaledW, scaledH, offsetX, offsetY);

                StretchDIBits(
                    memDC,
                    offsetX, offsetY, scaledW, scaledH,
                    0, 0, dib.width, dib.height,
                    dib.bgra.data(),
                    &dib.bmi,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
            } else if (isRectifiedView) {
                // Show "Top-Down unavailable" text for rectified view when not available
                SetBkMode(memDC, TRANSPARENT);
                SetTextColor(memDC, RGB(200, 200, 200));
                HFONT hOldFont = (HFONT)SelectObject(memDC, GetStockObject(DEFAULT_GUI_FONT));
                const wchar_t* text = L"Top-Down unavailable";
                RECT textRect = rc;
                DrawTextW(memDC, text, -1, &textRect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
                SelectObject(memDC, hOldFont);
            }

            BitBlt(hdc, 0, 0, dstW, dstH, memDC, 0, 0, SRCCOPY);
            SelectObject(memDC, oldBmp);
            DeleteObject(memBmp);
            DeleteDC(memDC);

            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_MOUSEMOVE: {
            // Handle color picker mouse move in ImageView
            if (g_colorPickerTarget != ColorPickerTarget::None && !g_lastSourceFrame.empty()) {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);
                
                // Enable continuous mouse tracking
                TRACKMOUSEEVENT tme = {};
                tme.cbSize = sizeof(TRACKMOUSEEVENT);
                tme.dwFlags = TME_HOVER | TME_LEAVE;
                tme.hwndTrack = hwnd;
                tme.dwHoverTime = 1;
                TrackMouseEvent(&tme);
                
                // Convert client coordinates to source image coordinates
                RECT rc{};
                GetClientRect(hwnd, &rc);
                const int dstW = std::max(0L, rc.right - rc.left);
                const int dstH = std::max(0L, rc.bottom - rc.top);
                
                int imgX, imgY;
                // Use g_imageDib dimensions for coordinate conversion (displayed image)
                // Account for aspect-ratio-preserving scaling
                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                    int scaledW, scaledH, offsetX, offsetY;
                    calculateScaledImageBounds(g_imageDib.width, g_imageDib.height, dstW, dstH, scaledW, scaledH, offsetX, offsetY);
                    
                    // Check if mouse is within the scaled image bounds
                    if (x >= offsetX && x < offsetX + scaledW && y >= offsetY && y < offsetY + scaledH) {
                        // Convert coordinates relative to the scaled image area
                        const int relX = x - offsetX;
                        const int relY = y - offsetY;
                        imgX = (int)((double)relX / scaledW * g_imageDib.width);
                        imgY = (int)((double)relY / scaledH * g_imageDib.height);
                        // Clamp to source frame dimensions
                        imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                        imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
                    } else {
                        // Mouse is outside the image area (in letterbox/pillarbox region)
                        imgX = -1;
                        imgY = -1;
                    }
                } else {
                    imgX = std::clamp(x, 0, g_lastSourceFrame.cols - 1);
                    imgY = std::clamp(y, 0, g_lastSourceFrame.rows - 1);
                }
                
                // Get screen coordinates for positioning zoom window
                POINT pt = {x, y};
                ClientToScreen(hwnd, &pt);
                
                // Only update magnifier if mouse is over the image
                if (imgX >= 0 && imgY >= 0) {
                    updateMagnifierWindow(imgX, imgY, pt.x, pt.y);
                }
            }
            return DefWindowProc(hwnd, msg, wParam, lParam);
        }
        case WM_MOUSELEAVE: {
            // Hide magnifier when mouse leaves the image view
            if (g_magnifierHwnd) {
                ShowWindow(g_magnifierHwnd, SW_HIDE);
            }
            break;
        }
        case WM_LBUTTONDOWN: {
            // Handle color picker click in ImageView
            if (g_colorPickerTarget != ColorPickerTarget::None && !g_lastSourceFrame.empty()) {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);
                
                // Convert client coordinates to source image coordinates
                RECT rc{};
                GetClientRect(hwnd, &rc);
                const int dstW = std::max(0L, rc.right - rc.left);
                const int dstH = std::max(0L, rc.bottom - rc.top);
                
                int imgX, imgY;
                // Use g_imageDib dimensions for coordinate conversion (displayed image)
                // Account for aspect-ratio-preserving scaling
                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                    int scaledW, scaledH, offsetX, offsetY;
                    calculateScaledImageBounds(g_imageDib.width, g_imageDib.height, dstW, dstH, scaledW, scaledH, offsetX, offsetY);
                    
                    // Check if mouse is within the scaled image bounds
                    if (x >= offsetX && x < offsetX + scaledW && y >= offsetY && y < offsetY + scaledH) {
                        // Convert coordinates relative to the scaled image area
                        const int relX = x - offsetX;
                        const int relY = y - offsetY;
                        imgX = (int)((double)relX / scaledW * g_imageDib.width);
                        imgY = (int)((double)relY / scaledH * g_imageDib.height);
                        // Clamp to source frame dimensions
                        imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                        imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
                    } else {
                        // Mouse is outside the image area (in letterbox/pillarbox region)
                        imgX = -1;
                        imgY = -1;
                    }
                } else {
                    imgX = std::clamp(x, 0, g_lastSourceFrame.cols - 1);
                    imgY = std::clamp(y, 0, g_lastSourceFrame.rows - 1);
                }
                
                // Sample color at click position
                if (imgX >= 0 && imgX < g_lastSourceFrame.cols && imgY >= 0 && imgY < g_lastSourceFrame.rows) {
                    cv::Vec3b bgr = g_lastSourceFrame.at<cv::Vec3b>(imgY, imgX);
                    
                    // Convert BGR to HSV
                    cv::Mat bgrMat(1, 1, CV_8UC3);
                    bgrMat.at<cv::Vec3b>(0, 0) = bgr;
                    cv::Mat hsvMat;
                    cv::cvtColor(bgrMat, hsvMat, cv::COLOR_BGR2HSV);
                    cv::Vec3b hsv = hsvMat.at<cv::Vec3b>(0, 0);
                    
                    if (g_colorPickerTarget == ColorPickerTarget::Felt) {
                        uiControls.feltParams.pickedBGR = bgr;
                        uiControls.feltParams.pickedHSV = hsv;
                        uiControls.feltParams.hasPickedColor = true;
                        applyFeltColorSensitivityToRangesFromPickedHSV();
                    } else if (g_colorPickerTarget == ColorPickerTarget::Rail) {
                        uiControls.railParams.pickedBGR = bgr;
                        uiControls.railParams.pickedHSV = hsv;
                        uiControls.railParams.hasPickedColor = true;
                        applyRailColorSensitivityToRangesFromPickedHSV();
                    }
                    
                    // Hide magnifier and deactivate picker FIRST, then refresh UI.
                    // `updateColorPickerLabels()` uses `g_colorPickerTarget` to decide the button titles.
                    if (g_magnifierHwnd) {
                        ShowWindow(g_magnifierHwnd, SW_HIDE);
                    }
                    g_colorPickerTarget = ColorPickerTarget::None;
                    
                    // Save settings immediately when color is picked
                    saveSettingsToDisk();

                    // Update UI labels (also updates button text)
                    updateColorPickerLabels();
                }
            }
            break;
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static void updateImageDibFromBgr(const cv::Mat& bgr) {
    if (bgr.empty()) return;
    if (bgr.type() != CV_8UC3) return;

    const int w = bgr.cols;
    const int h = bgr.rows;

    if (g_imageDib.width != w || g_imageDib.height != h) {
        g_imageDib = {};
        g_imageDib.width = w;
        g_imageDib.height = h;
        g_imageDib.bgra.resize((size_t)w * (size_t)h * 4);

        g_imageDib.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        g_imageDib.bmi.bmiHeader.biWidth = w;
        g_imageDib.bmi.bmiHeader.biHeight = -h; // top-down
        g_imageDib.bmi.bmiHeader.biPlanes = 1;
        g_imageDib.bmi.bmiHeader.biBitCount = 32;
        g_imageDib.bmi.bmiHeader.biCompression = BI_RGB;
        g_imageDib.bmi.bmiHeader.biSizeImage = 0;
    }

    // Convert BGR -> BGRA (alpha = 255)
    const int srcStride = (int)bgr.step;
    const unsigned char* src = bgr.data;
    unsigned char* dst = g_imageDib.bgra.data();
    for (int y = 0; y < h; ++y) {
        const unsigned char* s = src + y * srcStride;
        unsigned char* d = dst + (size_t)y * (size_t)w * 4;
        for (int x = 0; x < w; ++x) {
            d[x * 4 + 0] = s[x * 3 + 0]; // B
            d[x * 4 + 1] = s[x * 3 + 1]; // G
            d[x * 4 + 2] = s[x * 3 + 2]; // R
            d[x * 4 + 3] = 255;          // A
        }
    }
}

// Update rectified DIB buffer from BGR image
static void updateRectifiedDibFromBgr(const cv::Mat& bgr) {
    if (bgr.empty()) {
        // Clear the DIB if image is empty
        g_rectifiedDib = {};
        return;
    }
    if (bgr.type() != CV_8UC3) return;

    const int w = bgr.cols;
    const int h = bgr.rows;

    if (g_rectifiedDib.width != w || g_rectifiedDib.height != h) {
        g_rectifiedDib = {};
        g_rectifiedDib.width = w;
        g_rectifiedDib.height = h;
        g_rectifiedDib.bgra.resize((size_t)w * (size_t)h * 4);

        g_rectifiedDib.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        g_rectifiedDib.bmi.bmiHeader.biWidth = w;
        g_rectifiedDib.bmi.bmiHeader.biHeight = -h; // top-down
        g_rectifiedDib.bmi.bmiHeader.biPlanes = 1;
        g_rectifiedDib.bmi.bmiHeader.biBitCount = 32;
        g_rectifiedDib.bmi.bmiHeader.biCompression = BI_RGB;
        g_rectifiedDib.bmi.bmiHeader.biSizeImage = 0;
    }

    // Convert BGR -> BGRA (alpha = 255)
    const int srcStride = (int)bgr.step;
    const unsigned char* src = bgr.data;
    unsigned char* dst = g_rectifiedDib.bgra.data();
    for (int y = 0; y < h; ++y) {
        const unsigned char* s = src + y * srcStride;
        unsigned char* d = dst + (size_t)y * (size_t)w * 4;
        for (int x = 0; x < w; ++x) {
            d[x * 4 + 0] = s[x * 3 + 0]; // B
            d[x * 4 + 1] = s[x * 3 + 1]; // G
            d[x * 4 + 2] = s[x * 3 + 2]; // R
            d[x * 4 + 3] = 255;          // A
        }
    }
}

// MOVE?
// Build display frame for rectified (top-down) view with optional overlay
static cv::Mat buildRectifiedDisplayFrame() {
    if (!latestAnalysis.rect.ok || latestAnalysis.rect.rectifiedBgr.empty()) {
        // Return empty mat if rectification is not available
        return cv::Mat();
    }

    cv::Mat processed = latestAnalysis.rect.rectifiedBgr.clone();

    // Apply felt mask overlay if enabled (using felt_detection module)
    if (uiControls.showOverlay && uiControls.showFelt && !latestAnalysis.rect.rectifiedFeltMask.empty()) {
        drawFeltOverlayFromMask(processed, latestAnalysis.rect.rectifiedFeltMask, uiControls.feltParams);
    }

    // Draw felt corners quad on rectified view (transform corners from source to rectified space)
    if (uiControls.showOverlay && uiControls.showFelt && latestAnalysis.felt.hasCorners && !latestAnalysis.rect.H.empty()) {
        // Transform felt corners from source space to rectified space using homography
        std::vector<cv::Point2f> srcCornersF;
        for (const auto& corner : latestAnalysis.felt.corners) {
            srcCornersF.push_back(corner);
        }
        std::vector<cv::Point2f> transformedCorners;
        cv::perspectiveTransform(srcCornersF, transformedCorners, latestAnalysis.rect.H);
        
        // Draw the quad with darker felt color
        if (transformedCorners.size() == 4) {
            std::vector<cv::Point> quadPts;
            for (const auto& corner : transformedCorners) {
                quadPts.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
            }
            
            // Use a darker version of the felt overlay color (multiply each channel by 0.7)
            cv::Scalar darkerColor(
                uiControls.feltParams.color[0] * 0.7,
                uiControls.feltParams.color[1] * 0.7,
                uiControls.feltParams.color[2] * 0.7
            );
            
            cv::polylines(processed, std::vector<std::vector<cv::Point>>{quadPts}, true, darkerColor, 2);
        }
    }

    // Apply rail overlay if enabled (using rail_detection module)
    if (uiControls.showOverlay && uiControls.showRails && latestAnalysis.rails.ok) {
        drawRailOverlay(processed, latestAnalysis.rails, uiControls.railParams);
    }

    // Apply diamond overlay if enabled (using diamond_detection module)
    if (uiControls.showOverlay && uiControls.showDiamonds && latestAnalysis.diamonds.ok) {
        drawDiamondOverlay(processed, latestAnalysis.diamonds, uiControls.diamondParams);
    }

    return processed;
}

static void ensureSidebarAndImageChildren(HWND mainHwnd) {
    if (!g_imageViewHwnd) {
        g_imageViewHwnd = CreateWindowExW(
            0,
            kImageViewClass,
            L"",
            WS_VISIBLE | WS_CHILD,
            0, 0, 100, 100,
            mainHwnd,
            (HMENU)(INT_PTR)(10000 + 400),
            GetModuleHandleW(NULL),
            NULL
        );
    }

    if (!g_rectifiedViewHwnd) {
        g_rectifiedViewHwnd = CreateWindowExW(
            0,
            kImageViewClass,
            L"",
            WS_VISIBLE | WS_CHILD,
            0, 0, 100, 100,
            mainHwnd,
            (HMENU)(INT_PTR)(10000 + 401),
            GetModuleHandleW(NULL),
            NULL
        );
    }

    if (!g_sidebarPanel) {
        if (!g_sidebarBgBrush) {
            g_sidebarBgBrush = CreateSolidBrush(RGB(240, 240, 240));
        }
        g_sidebarPanel = CreateWindowExW(
            0,
            kSidebarPanelClass,
            L"",
            WS_VISIBLE | WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VSCROLL,
            0, 0, 300, 100,
            mainHwnd,
            (HMENU)(INT_PTR)(10000 + 200),
            GetModuleHandleW(NULL),
            NULL
        );
    }
}

// Update color picker info labels and swatch in the UI
static void updateColorPickerLabels() {
    if (!g_sidebarPanel || !IsWindow(g_sidebarPanel)) return;

    auto formatRange = [](int hMin, int hMax, int sMin, int sMax, int vMin, int vMax, wchar_t* out, size_t outCount) {
        if (hMin <= hMax) {
            swprintf_s(out, outCount, L"Range: H[%d-%d] S[%d-%d] V[%d-%d]", hMin, hMax, sMin, sMax, vMin, vMax);
        } else {
            swprintf_s(out, outCount, L"Range: H[0-%d] U [%d-180] S[%d-%d] V[%d-%d]", hMax, hMin, sMin, sMax, vMin, vMax);
        }
    };

    // --------------------------
    // Felt picker labels/swatch
    // --------------------------
    {
        wchar_t bgrText[64] = L"BGR: --";
        wchar_t hsvText[64] = L"HSV: --";
        wchar_t rangeText[128] = L"Range: --";

        const cv::Vec3b bgr = uiControls.feltParams.pickedBGR;
        swprintf_s(bgrText, L"BGR: (%d, %d, %d)", (int)bgr[2], (int)bgr[1], (int)bgr[0]);

        if (uiControls.feltParams.hasPickedColor) {
            const cv::Vec3b hsv = uiControls.feltParams.pickedHSV;
            swprintf_s(hsvText, L"HSV: (%d, %d, %d)", (int)hsv[0], (int)hsv[1], (int)hsv[2]);
        }

        formatRange(uiControls.feltParams.colorHMin, uiControls.feltParams.colorHMax,
                    uiControls.feltParams.colorSMin, uiControls.feltParams.colorSMax,
                    uiControls.feltParams.colorVMin, uiControls.feltParams.colorVMax,
                    rangeText, _countof(rangeText));

        HWND hBGR = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_PICKER_BGR);
        if (hBGR) SetWindowTextW(hBGR, bgrText);
        HWND hHSV = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_PICKER_HSV);
        if (hHSV) SetWindowTextW(hHSV, hsvText);
        HWND hRange = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_PICKER_RANGE);
        if (hRange) SetWindowTextW(hRange, rangeText);

        HWND hSwatch = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_PICKER_SWATCH);
        if (hSwatch) {
            const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]);
            setSwatchBrush(g_feltColorPickerSwatchBrush, swatchColor);
            InvalidateRect(hSwatch, NULL, TRUE);
            UpdateWindow(hSwatch);
        }

        HWND hColorPicker = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_PICKER);
        if (hColorPicker) {
            const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Felt);
            const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Felt Color";
            SetWindowTextW(hColorPicker, btnText);
        }
    }

    // --------------------------
    // Diamond picker labels/swatch (only update if section is expanded)
    // --------------------------
    if (uiControls.diamondsExpanded) {
        wchar_t bgrText[64] = L"BGR: --";
        wchar_t hsvText[64] = L"HSV: --";
        wchar_t rangeText[128] = L"Range: --";

        const cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
        swprintf_s(bgrText, L"BGR: (%d, %d, %d)", (int)bgr[2], (int)bgr[1], (int)bgr[0]);

        if (uiControls.diamondParams.hasPickedColor) {
            const cv::Vec3b hsv = uiControls.diamondParams.pickedHSV;
            swprintf_s(hsvText, L"HSV: (%d, %d, %d)", (int)hsv[0], (int)hsv[1], (int)hsv[2]);
        }

        formatRange(uiControls.diamondParams.colorHMin, uiControls.diamondParams.colorHMax,
                    uiControls.diamondParams.colorSMin, uiControls.diamondParams.colorSMax,
                    uiControls.diamondParams.colorVMin, uiControls.diamondParams.colorVMax,
                    rangeText, _countof(rangeText));

        HWND hBGR = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER_BGR);
        if (hBGR) SetWindowTextW(hBGR, bgrText);
        HWND hHSV = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER_HSV);
        if (hHSV) SetWindowTextW(hHSV, hsvText);
        HWND hRange = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER_RANGE);
        if (hRange) SetWindowTextW(hRange, rangeText);

        HWND hSwatch = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER_SWATCH);
        if (hSwatch) {
            const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]);
            setSwatchBrush(g_diamondColorPickerSwatchBrush, swatchColor);
            InvalidateRect(hSwatch, NULL, TRUE);
            UpdateWindow(hSwatch);
        }

        HWND hColorPicker = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER);
        if (hColorPicker) {
            const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Diamond);
            const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Diamond Color";
            SetWindowTextW(hColorPicker, btnText);
        }
    }

    // --------------------------
    // Rail picker labels/swatch (only update if section is expanded)
    // --------------------------
    if (uiControls.railsExpanded) {
        wchar_t bgrText[64] = L"BGR: --";
        wchar_t hsvText[64] = L"HSV: --";
        wchar_t rangeText[128] = L"Range: --";

        const cv::Vec3b bgr = uiControls.railParams.pickedBGR;
        swprintf_s(bgrText, L"BGR: (%d, %d, %d)", (int)bgr[2], (int)bgr[1], (int)bgr[0]);

        if (uiControls.railParams.hasPickedColor) {
            const cv::Vec3b hsv = uiControls.railParams.pickedHSV;
            swprintf_s(hsvText, L"HSV: (%d, %d, %d)", (int)hsv[0], (int)hsv[1], (int)hsv[2]);
        }

        formatRange(uiControls.railParams.colorHMin, uiControls.railParams.colorHMax,
                    uiControls.railParams.colorSMin, uiControls.railParams.colorSMax,
                    uiControls.railParams.colorVMin, uiControls.railParams.colorVMax,
                    rangeText, _countof(rangeText));

        HWND hBGR = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_PICKER_BGR);
        if (hBGR) SetWindowTextW(hBGR, bgrText);
        HWND hHSV = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_PICKER_HSV);
        if (hHSV) SetWindowTextW(hHSV, hsvText);
        HWND hRange = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_PICKER_RANGE);
        if (hRange) SetWindowTextW(hRange, rangeText);

        HWND hSwatch = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_PICKER_SWATCH);
        if (hSwatch) {
            const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]);
            setSwatchBrush(g_railColorPickerSwatchBrush, swatchColor);
            InvalidateRect(hSwatch, NULL, TRUE);
            UpdateWindow(hSwatch);
        }

        HWND hColorPicker = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_PICKER);
        if (hColorPicker) {
            const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Rail);
            const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Rail Color";
            SetWindowTextW(hColorPicker, btnText);
        }
    }
}

// Shared HSV tolerance computation based on the user-facing sensitivity (0..100).
// - 0   => tight match (strict)
// - 100 => loose match (more variation)
static void computeHsvRangeFromPickedHsv(
    const cv::Vec3b& pickedHSV,
    int sensitivity,
    int& outHMin,
    int& outHMax,
    int& outSMin,
    int& outSMax,
    int& outVMin,
    int& outVMax
) {
    const int sens = std::clamp(sensitivity, 0, 100);
    const float t = static_cast<float>(sens) / 100.0f;

    // Tuned to be usable for typical consumer cameras / mixed lighting.
    // Hue is 0..180 in OpenCV. Saturation/Value are 0..255.
    const int hTol = (int)std::lround(5.0f + t * 25.0f);    // 5..30
    const int sTol = (int)std::lround(10.0f + t * 80.0f);   // 10..90
    const int vTol = (int)std::lround(10.0f + t * 100.0f);  // 10..110

    const int h = (int)pickedHSV[0];
    const int s = (int)pickedHSV[1];
    const int v = (int)pickedHSV[2];

    // Hue wrap: represent wrap by setting HMin > HMax (consumer of range must handle wrap).
    const int hMinRaw = h - hTol;
    const int hMaxRaw = h + hTol;
    if (hMinRaw < 0 || hMaxRaw > 180) {
        const int hMinWrap = (hMinRaw < 0) ? (180 + hMinRaw) : hMinRaw;
        const int hMaxWrap = (hMaxRaw > 180) ? (hMaxRaw - 180) : hMaxRaw;
        outHMin = std::clamp(hMinWrap, 0, 180);
        outHMax = std::clamp(hMaxWrap, 0, 180);
    } else {
        outHMin = std::clamp(hMinRaw, 0, 180);
        outHMax = std::clamp(hMaxRaw, 0, 180);
    }

    outSMin = std::clamp(s - sTol, 0, 255);
    outSMax = std::clamp(s + sTol, 0, 255);
    outVMin = std::clamp(v - vTol, 0, 255);
    outVMax = std::clamp(v + vTol, 0, 255);
}

// Felt-specific HSV tolerance:
// Shadows are the dominant failure mode for felt segmentation (V drops substantially and S can soften),
// so we intentionally allow a *much larger* tolerance toward darker values than toward brighter values.
// This helps keep the felt mask connected even under uneven lighting.
static void computeFeltHsvRangeFromPickedHsv(
    const cv::Vec3b& pickedHSV,
    int sensitivity,
    int& outHMin,
    int& outHMax,
    int& outSMin,
    int& outSMax,
    int& outVMin,
    int& outVMax
) {
    const int sens = std::clamp(sensitivity, 0, 100);
    const float t = static_cast<float>(sens) / 100.0f;

    // Hue stays relatively stable under shadows; keep H tighter to avoid grabbing non-felt.
    const int hTol = (int)std::lround(4.0f + t * 18.0f);     // 4..22

    // Saturation can drift down under shadows / glare; allow more variation than diamonds.
    const int sTol = (int)std::lround(20.0f + t * 120.0f);   // 20..140

    // Value tolerance is intentionally asymmetric:
    // - allow a lot darker (shadows)
    // - allow a bit brighter (hotspots)
    const int vTolDown = (int)std::lround(40.0f + t * 160.0f); // 40..200
    const int vTolUp   = (int)std::lround(10.0f + t * 60.0f);  // 10..70

    const int h = (int)pickedHSV[0];
    const int s = (int)pickedHSV[1];
    const int v = (int)pickedHSV[2];

    // Hue wrap: represent wrap by setting HMin > HMax (consumer of range must handle wrap).
    const int hMinRaw = h - hTol;
    const int hMaxRaw = h + hTol;
    if (hMinRaw < 0 || hMaxRaw > 180) {
        const int hMinWrap = (hMinRaw < 0) ? (180 + hMinRaw) : hMinRaw;
        const int hMaxWrap = (hMaxRaw > 180) ? (hMaxRaw - 180) : hMaxRaw;
        outHMin = std::clamp(hMinWrap, 0, 180);
        outHMax = std::clamp(hMaxWrap, 0, 180);
    } else {
        outHMin = std::clamp(hMinRaw, 0, 180);
        outHMax = std::clamp(hMaxRaw, 0, 180);
    }

    outSMin = std::clamp(s - sTol, 0, 255);
    outSMax = std::clamp(s + sTol, 0, 255);
    outVMin = std::clamp(v - vTolDown, 0, 255);
    outVMax = std::clamp(v + vTolUp, 0, 255);
}

// Apply diamond color sensitivity to HSV ranges (similar to felt/rail)
static void applyDiamondColorSensitivityToRangesFromPickedHSV() {
    if (!uiControls.diamondParams.hasPickedColor) return;

    computeHsvRangeFromPickedHsv(
        uiControls.diamondParams.pickedHSV,
        uiControls.diamondParams.colorSensitivity,
        uiControls.diamondParams.colorHMin,
        uiControls.diamondParams.colorHMax,
        uiControls.diamondParams.colorSMin,
        uiControls.diamondParams.colorSMax,
        uiControls.diamondParams.colorVMin,
        uiControls.diamondParams.colorVMax
    );

    // Enable color filtering automatically when color is picked
    uiControls.diamondParams.useColorFilter = true;
}

// MOVE?
static void applyFeltColorSensitivityToRangesFromPickedHSV() {
    if (!uiControls.feltParams.hasPickedColor) return;
    computeFeltHsvRangeFromPickedHsv(
        uiControls.feltParams.pickedHSV,
        uiControls.feltParams.colorSensitivity,
        uiControls.feltParams.colorHMin,
        uiControls.feltParams.colorHMax,
        uiControls.feltParams.colorSMin,
        uiControls.feltParams.colorSMax,
        uiControls.feltParams.colorVMin,
        uiControls.feltParams.colorVMax
    );
}

// Apply rail color sensitivity to HSV ranges (similar to felt)
static void applyRailColorSensitivityToRangesFromPickedHSV() {
    if (!uiControls.railParams.hasPickedColor) return;

    computeHsvRangeFromPickedHsv(
        uiControls.railParams.pickedHSV,
        uiControls.railParams.colorSensitivity,
        uiControls.railParams.colorHMin,
        uiControls.railParams.colorHMax,
        uiControls.railParams.colorSMin,
        uiControls.railParams.colorSMax,
        uiControls.railParams.colorVMin,
        uiControls.railParams.colorVMax
    );

    // Enable color filtering automatically when color is picked
    uiControls.railParams.useColorFilter = true;

    // Note: railColorMin/Max in BGR format are not used anymore
    // The detection now uses the HSV ranges directly
}

static void layoutChildren(HWND mainHwnd) {
    RECT rc{};
    GetClientRect(mainHwnd, &rc);
    const int w = (int)(rc.right - rc.left);
    const int h = (int)(rc.bottom - rc.top);

    const int sidebarW =
        (uiControls.showSidebar && !uiControls.sidebarCollapsed) ? 300 :
        (uiControls.showSidebar && uiControls.sidebarCollapsed) ? 30 : 0;

    // Split available width between two viewports (Live View and Top-Down View)
    const int availableW = std::max(0, w - sidebarW);
    const int imageH = std::max(0, h);
    
    // Each viewport gets half the available width
    const int viewportW = availableW / 2;
    const int liveViewW = viewportW;
    const int topDownViewW = availableW - viewportW;  // Remaining width goes to top-down view

    // Position Live View (left viewport)
    if (g_imageViewHwnd) {
        MoveWindow(g_imageViewHwnd, 0, 0, liveViewW, imageH, TRUE);
    }
    
    // Position Top-Down View (middle viewport)
    if (g_rectifiedViewHwnd) {
        MoveWindow(g_rectifiedViewHwnd, liveViewW, 0, topDownViewW, imageH, TRUE);
    }
    
    // Position Sidebar (right column)
    if (g_sidebarPanel) {
        if (uiControls.showSidebar) {
            ShowWindow(g_sidebarPanel, SW_SHOW);
            MoveWindow(g_sidebarPanel, availableW, 0, sidebarW, imageH, TRUE);
        } else {
            ShowWindow(g_sidebarPanel, SW_HIDE);
        }
    }
}

// Minimal per-frame render: apply overlays (using OpenCV), then push to ImageView.
// We keep a small amount of global state to support temporal smoothing.
static cv::Mat g_prevOverlayDeltaF; // CV_32FC3: (processed - rawFrame) from previous frame
static std::array<cv::Point2f, 4> g_smoothedCorners; // Temporally smoothed corners for stable rectification
static bool g_cornersInitialized = false;
static cv::Mat g_prevRailMask; // Previous frame's rail mask for temporal smoothing
static bool g_railMaskInitialized = false;

// MOVE?
static cv::Mat buildDisplayFrame(const cv::Mat& currentFrame) {
    if (currentFrame.empty()) return {};
    cv::Mat processed = currentFrame.clone();

    // Felt detection - always run for rectification, regardless of showFelt toggle
    // showFelt only controls whether the overlay is displayed
    g_lastFeltResult = detectFelt(currentFrame, uiControls.feltParams);


    if (uiControls.showOverlay) {
        // Apply felt overlay only if showFelt is enabled (using felt_detection module)
        if (uiControls.showFelt) {
            drawFeltOverlay(processed, g_lastFeltResult, uiControls.feltParams);
            // Draw felt corners quad (from felt_detection module) - darker version of felt color
            drawFeltCornersQuad(processed, g_lastFeltResult, uiControls.feltParams);
        }
        
        // Apply rail overlay only if showRails is enabled (using rail_detection module)
        if (uiControls.showRails && latestAnalysis.railsPerspective.ok) {
            drawRailOverlay(processed, latestAnalysis.railsPerspective, uiControls.railParams);
        }

        // Apply diamond overlay if enabled (using diamond_detection module)
        if (uiControls.showDiamonds && !latestAnalysis.diamondsPerspective.diamonds.empty()) {
            drawDiamondOverlay(processed, latestAnalysis.diamondsPerspective, uiControls.diamondParams);
        }
    }

    // Store felt result in latestAnalysis
    latestAnalysis.felt = g_lastFeltResult;

    // Perform rectification if felt detection succeeded and has corners
    if (latestAnalysis.felt.ok && latestAnalysis.felt.hasCorners) {
        // Apply temporal smoothing to corners for stable rectification
        // This uses the smoothingPercent slider to control corner jitter in the rectified view
        const int sPct = std::clamp(uiControls.smoothingPercent, 0, 100);
        std::array<cv::Point2f, 4> cornersToUse = latestAnalysis.felt.corners;

        if (sPct > 0) {
            const float alpha = static_cast<float>(sPct) / 100.0f;
            if (!g_cornersInitialized) {
                g_smoothedCorners = latestAnalysis.felt.corners;
                g_cornersInitialized = true;
            } else {
                // Exponential moving average smoothing
                for (size_t i = 0; i < 4; ++i) {
                    g_smoothedCorners[i].x = alpha * g_smoothedCorners[i].x + (1.0f - alpha) * latestAnalysis.felt.corners[i].x;
                    g_smoothedCorners[i].y = alpha * g_smoothedCorners[i].y + (1.0f - alpha) * latestAnalysis.felt.corners[i].y;
                }
            }
            cornersToUse = g_smoothedCorners;
        }

        // Call rectification with smoothed corners
        // rectifyTabletop is now configured to output 2:1 portrait aspect ratio
        latestAnalysis.rect = rectifyTabletop(
            currentFrame,
            latestAnalysis.felt.feltMask,
            cornersToUse,
            uiControls.rectifyMarginScale,
            uiControls.rectifyPadPx
        );

    } else {
        // Reset rectification results if felt detection failed
        latestAnalysis.rect = RectificationResult();
        g_cornersInitialized = false;  // Reset smoothing when corners are lost
    }

    // Perform rail detection on rectified view if rectification succeeded
    if (latestAnalysis.rect.ok && !latestAnalysis.rect.rectifiedBgr.empty() && !latestAnalysis.rect.rectifiedFeltMask.empty()) {
        latestAnalysis.rails = detectRails(
            latestAnalysis.rect.rectifiedBgr,
            latestAnalysis.rect.rectifiedFeltMask,
            uiControls.railParams
        );
        
        // Apply temporal smoothing to rail mask to reduce jitter
        // Use a higher smoothing factor for rails to reduce intermittent detection
        const int sPct = std::clamp(uiControls.smoothingPercent, 0, 100);
        if (sPct > 0 && latestAnalysis.rails.ok && !latestAnalysis.rails.railMask.empty()) {
            // Apply stronger smoothing for rails (1.5x the slider value, clamped to max 95%)
            const float boostedAlpha = std::min(0.95f, static_cast<float>(sPct) * 1.5f / 100.0f);

            if (!g_railMaskInitialized || g_prevRailMask.size() != latestAnalysis.rails.railMask.size()) {
                latestAnalysis.rails.railMask.convertTo(g_prevRailMask, CV_32FC1, 1.0 / 255.0);
                g_railMaskInitialized = true;
            } else {
                // Exponential moving average smoothing on rail mask
                cv::Mat currentMaskF;
                latestAnalysis.rails.railMask.convertTo(currentMaskF, CV_32FC1, 1.0 / 255.0);
                cv::Mat smoothedMaskF;
                cv::addWeighted(currentMaskF, 1.0f - boostedAlpha, g_prevRailMask, boostedAlpha, 0.0, smoothedMaskF);

                // Use a higher threshold (0.6 instead of 0.5) to reduce noise and flickering
                // This provides hysteresis - rails must be consistently detected to appear
                cv::threshold(smoothedMaskF, smoothedMaskF, 0.6, 1.0, cv::THRESH_BINARY);
                smoothedMaskF.convertTo(latestAnalysis.rails.railMask, CV_8UC1, 255.0);
                g_prevRailMask = smoothedMaskF;
            }
        } else if (latestAnalysis.rails.ok && !latestAnalysis.rails.railMask.empty()) {
            // Initialize or reset smoothing state
            if (g_prevRailMask.empty() || latestAnalysis.rails.railMask.size() != g_prevRailMask.size()) {
                latestAnalysis.rails.railMask.convertTo(g_prevRailMask, CV_32FC1, 1.0 / 255.0);
                g_railMaskInitialized = true;
            }
        } else {
            g_railMaskInitialized = false;
        }
        
        // Project rails to perspective view for display on main view
        if (latestAnalysis.rails.ok && !latestAnalysis.rect.Hinv.empty()) {
            latestAnalysis.railsPerspective = projectRailsToPerspective(
                latestAnalysis.rails,
                latestAnalysis.rect.Hinv,
                currentFrame.size()
            );
        } else {
            latestAnalysis.railsPerspective = RailDetectionResult();
        }
    } else {
        latestAnalysis.rails = RailDetectionResult();
        latestAnalysis.railsPerspective = RailDetectionResult();
        g_railMaskInitialized = false;
    }

    // Perform diamond detection on rectified view if rail detection succeeded
    if (latestAnalysis.rails.ok && !latestAnalysis.rails.railMasks.empty()) {
        latestAnalysis.diamonds = detectDiamonds(
            latestAnalysis.rect.rectifiedBgr,
            latestAnalysis.rails.railMasks,
            uiControls.diamondParams
        );

        // Project diamonds to perspective view for display on main view
        if (latestAnalysis.diamonds.ok && !latestAnalysis.rect.Hinv.empty()) {
            latestAnalysis.diamondsPerspective = projectDiamondsToPerspective(
                latestAnalysis.diamonds,
                latestAnalysis.rect.Hinv,
                currentFrame.size()
            );
        } else {
            latestAnalysis.diamondsPerspective = DiamondDetectionResult();
        }
    } else {
        latestAnalysis.diamonds = DiamondDetectionResult();
        latestAnalysis.diamondsPerspective = DiamondDetectionResult();
    }

    // --------------------------------------------------------------------------------------------
    // Global temporal smoothing (stabilize overlays without blurring the base video)
    //
    // Approach:
    // - `processed` is the output image after overlays are drawn.
    // - We compute delta = processed - currentFrame (in float space).
    // - We exponentially smooth delta over time:
    //     smoothedDelta = (1 - s) * delta + s * prevDelta
    // - Final output = currentFrame + smoothedDelta
    //
    // This avoids "ghosting" the raw video while still stabilizing the overlay motion/jitter.
    // --------------------------------------------------------------------------------------------
    const int sPct = std::clamp(uiControls.smoothingPercent, 0, 100);
    if (sPct <= 0) {
        g_prevOverlayDeltaF.release();
        return processed;
    }

    const float s = static_cast<float>(sPct) / 100.0f;

    cv::Mat frameF;
    cv::Mat processedF;
    currentFrame.convertTo(frameF, CV_32FC3);
    processed.convertTo(processedF, CV_32FC3);

    cv::Mat deltaF = processedF - frameF;
    if (g_prevOverlayDeltaF.empty() || g_prevOverlayDeltaF.size() != deltaF.size() || g_prevOverlayDeltaF.type() != deltaF.type()) {
        g_prevOverlayDeltaF = deltaF.clone();
    }

    cv::Mat smoothedDeltaF;
    cv::addWeighted(deltaF, 1.0f - s, g_prevOverlayDeltaF, s, 0.0, smoothedDeltaF);
    g_prevOverlayDeltaF = smoothedDeltaF;

    cv::Mat outF = frameF + smoothedDeltaF;
    cv::Mat outU8;
    outF.convertTo(outU8, CV_8UC3);
    return outU8;
}

static void openSelectedCameraIfNeeded() {
    // Camera opening is handled by the background capture thread now.
    // Keep this function around (it is referenced by older code) but make it a no-op.
}

// Timer-driven updates (no OpenCV windowing).
static void onFrameTick(HWND mainHwnd) {
    cv::Mat frame;

    // Always render test image immediately when selected (never wait for capture I/O).
    if (uiControls.selectedSource == kSourceTestImage) {
        frame = g_testImage;
        g_requestedCaptureSource.store(kSourceTestImage, std::memory_order_relaxed);
    } else {
        g_requestedCaptureSource.store(uiControls.selectedSource, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(g_latestFrameMutex);
            frame = g_latestCaptureFrame;
        }
        if (frame.empty()) {
            frame = g_testImage;
        }
    }

    // Keep source frame for color picking / magnifier.
    // NOTE: In the Win32-hosted UI path, the picker reads pixels from `g_lastSourceFrame`.
    // Without updating this each tick, the picker appears "dead" (no magnifier, no picked color updates).
    g_lastSourceFrame = frame.clone();

    cv::Mat display = buildDisplayFrame(frame);
    if (!display.empty()) {
        // Keep a copy of the most recent rendered frame so "Export Captures" works in Win32 mode
        // (this code path does not use the OpenCV `cv::imshow` loop).
        g_lastProcessedFrame = display.clone();

        updateImageDibFromBgr(display);
        if (g_imageViewHwnd) {
            // Avoid erase flicker; ImageView fully repaints.
            RedrawWindow(g_imageViewHwnd, NULL, NULL, RDW_INVALIDATE | RDW_NOERASE);
        }
    }

    // Update rectified (top-down) view
    cv::Mat rectifiedDisplay = buildRectifiedDisplayFrame();
    if (!rectifiedDisplay.empty()) {
        updateRectifiedDibFromBgr(rectifiedDisplay);
        if (g_rectifiedViewHwnd) {
            RedrawWindow(g_rectifiedViewHwnd, NULL, NULL, RDW_INVALIDATE | RDW_NOERASE);
        }
    } else {
        // Clear rectified view if not available
        g_rectifiedDib = {};
        if (g_rectifiedViewHwnd) {
            RedrawWindow(g_rectifiedViewHwnd, NULL, NULL, RDW_INVALIDATE | RDW_NOERASE);
        }
    }
}

static LRESULT CALLBACK MainWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            g_hwnd = hwnd;

            // Ensure the window title is set using the wide API (avoids ANSI/Unicode truncation).
            SetWindowTextW(hwnd, L"Billiards Trainer");

            // Ensure common controls (trackbars) work.
            INITCOMMONCONTROLSEX icc{};
            icc.dwSize = sizeof(icc);
            icc.dwICC = ICC_BAR_CLASSES;
            InitCommonControlsEx(&icc);

            // Sidebar font styling (modern-ish default on Windows)
            // Segoe UI is the standard UI font on modern Windows.
            if (!g_sidebarFont || !g_sidebarFontBold) {
                HDC hdc = GetDC(hwnd);
                const int dpiY = GetDeviceCaps(hdc, LOGPIXELSY);
                ReleaseDC(hwnd, hdc);

                const int pt10 = -MulDiv(10, dpiY, 72);
                const int pt11 = -MulDiv(11, dpiY, 72);

                g_sidebarFont = CreateFontW(pt10, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
                g_sidebarFontBold = CreateFontW(pt11, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                                                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                                CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
            }

            createNativeMenu(hwnd, g_availableCameras);

            ensureSidebarAndImageChildren(hwnd);
            layoutChildren(hwnd);

            // Create sidebar controls inside g_sidebarPanel
            if (uiControls.showSidebar) {
                destroySidebarControls();
                createSidebarControls(hwnd);
            }

            // Drive frame updates at ~30fps
            SetTimer(hwnd, 1, 33, NULL);
            return 0;
        }
        case WM_SIZE: {
            ensureSidebarAndImageChildren(hwnd);
            layoutChildren(hwnd);
            if (uiControls.showSidebar) {
                destroySidebarControls();
                createSidebarControls(hwnd);
            }
            return 0;
        }
        case WM_TIMER: {
            if (wParam == 1) {
                onFrameTick(hwnd);
            }
            return 0;
        }
        case WM_COMMAND: {
            const int id = LOWORD(wParam);
            const int code = HIWORD(wParam);
            HWND hwndCtl = (HWND)lParam;

            // IMPORTANT:
            // WM_COMMAND is used for BOTH menu items (lParam == NULL) and control notifications (lParam != NULL).
            // We must NOT treat control notifications (like combobox dropdown/open) as menu commands,
            // otherwise we'll rebuild/destroy controls while they are interacting (causing crashes).

            // Sidebar buttons are regular WM_COMMANDs too.
            // NOTE: This Win32-hosted block appears before the later #define ID block in the file,
            // so we intentionally use literal IDs here (matches the defines below).
            const int kIdSidebarCollapse = 30100; // IDC_BUTTON_BASE(30000) + 100
            const int kIdSidebarDiamonds = 30101;
            const int kIdSidebarFelt = 30102;
            const int kIdSidebarRail = 30103;
            const int kIdSidebarRectify = 30104;
            if (id == kIdSidebarCollapse || id == kIdSidebarDiamonds || id == kIdSidebarFelt || id == kIdSidebarRail || id == kIdSidebarRectify) {
                // Reuse existing handler
                handleSidebarButton(id);
                layoutChildren(hwnd);
                destroySidebarControls();
                createSidebarControls(hwnd);
                return 0;
            }

            // Debug sidebar checkboxes (IDs match defines below: 30200..30203)
            const int kIdOverlayMaster = 30200;
            const int kIdOverlayDiamonds = 30201;
            const int kIdOverlayFelt = 30202;
            const int kIdOverlayRails = 30203;
            if (code == BN_CLICKED && (id == kIdOverlayMaster || id == kIdOverlayDiamonds || id == kIdOverlayFelt || id == kIdOverlayRails)) {
                const bool isChecked = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                if (id == kIdOverlayMaster) uiControls.showOverlay = isChecked;
                else if (id == kIdOverlayDiamonds) uiControls.showDiamonds = isChecked;
                else if (id == kIdOverlayFelt) uiControls.showFelt = isChecked;
                else if (id == kIdOverlayRails) uiControls.showRails = isChecked;
                saveSettingsToDisk();  // Save settings immediately when overlay toggles change
                return 0;
            }

            // Overlay style buttons/checkboxes (IDs match defines below: 30230+)
            const int kIdDiamondsColor = 30220;
            const int kIdDiamondsFilled = 30221;
            const int kIdFeltColor = 30230;
            const int kIdFeltFilled = 30231;
            const int kIdRailsColor = 30240;
            const int kIdRailsFilled = 30241;
            if (code == BN_CLICKED) {
                if (id == kIdDiamondsFilled) {
                    uiControls.diamondParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    saveSettingsToDisk();  // Save settings immediately when checkbox toggles
                    return 0;
                }
                if (id == kIdFeltFilled) {
                    uiControls.feltParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    saveSettingsToDisk();  // Save settings immediately when checkbox toggles
                    return 0;
                }
                if (id == kIdRailsFilled) {
                    uiControls.railParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    saveSettingsToDisk();  // Save settings immediately when checkbox toggles
                    return 0;
                }
                if (id == IDC_DIAMOND_COLOR_PICKER) {
                    // Activate/deactivate diamond color picker mode.
                    const ColorPickerTarget requested = ColorPickerTarget::Diamond;

                    if (g_colorPickerTarget == requested) {
                        // Deactivate
                        g_colorPickerTarget = ColorPickerTarget::None;
                        if (g_magnifierHwnd) ShowWindow(g_magnifierHwnd, SW_HIDE);
                    } else {
                        // Switch to requested picker
                        g_colorPickerTarget = requested;

                        // Enable mouse tracking in ImageView to get mouse move events
                        if (g_imageViewHwnd) {
                            TRACKMOUSEEVENT tme = {};
                            tme.cbSize = sizeof(TRACKMOUSEEVENT);
                            tme.dwFlags = TME_HOVER | TME_LEAVE;
                            tme.hwndTrack = g_imageViewHwnd;
                            tme.dwHoverTime = 1;  // Immediate hover
                            TrackMouseEvent(&tme);

                            // Get current mouse position and show magnifier immediately
                            POINT pt;
                            GetCursorPos(&pt);
                            ScreenToClient(g_imageViewHwnd, &pt);

                            // Convert to image coordinates and show magnifier
                            if (!g_lastSourceFrame.empty()) {
                                RECT rc;
                                GetClientRect(g_imageViewHwnd, &rc);
                                const int dstW = std::max(0L, rc.right - rc.left);
                                const int dstH = std::max(0L, rc.bottom - rc.top);

                                int imgX, imgY;
                                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                                    imgX = (int)((double)pt.x / dstW * g_imageDib.width);
                                    imgY = (int)((double)pt.y / dstH * g_imageDib.height);
                                    imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                                    imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);

                                    POINT screenPt = {pt.x, pt.y};
                                    ClientToScreen(g_imageViewHwnd, &screenPt);
                                    updateMagnifierWindow(imgX, imgY, screenPt.x, screenPt.y);
                                }
                            }
                        }
                    }
                    updateColorPickerLabels();  // Update button text and UI
                    return 0;
                }
                if (id == IDC_FELT_COLOR_PICKER) {
                    // Activate/deactivate color picker mode.
                    //
                    // Only one picker can be active at a time; the active one is identified by `g_colorPickerTarget`.
                    const ColorPickerTarget requested = ColorPickerTarget::Felt;

                    if (g_colorPickerTarget == requested) {
                        // Deactivate
                        g_colorPickerTarget = ColorPickerTarget::None;
                        if (g_magnifierHwnd) ShowWindow(g_magnifierHwnd, SW_HIDE);
                    } else {
                        // Switch to requested picker
                        g_colorPickerTarget = requested;

                        // Enable mouse tracking in ImageView to get mouse move events
                        if (g_imageViewHwnd) {
                            TRACKMOUSEEVENT tme = {};
                            tme.cbSize = sizeof(TRACKMOUSEEVENT);
                            tme.dwFlags = TME_HOVER | TME_LEAVE;
                            tme.hwndTrack = g_imageViewHwnd;
                            tme.dwHoverTime = 1;  // Immediate hover
                            TrackMouseEvent(&tme);
                            
                            // Get current mouse position and show magnifier immediately
                            POINT pt;
                            GetCursorPos(&pt);
                            ScreenToClient(g_imageViewHwnd, &pt);
                            
                            // Convert to image coordinates and show magnifier
                            if (!g_lastSourceFrame.empty()) {
                                RECT rc;
                                GetClientRect(g_imageViewHwnd, &rc);
                                const int dstW = std::max(0L, rc.right - rc.left);
                                const int dstH = std::max(0L, rc.bottom - rc.top);
                                
                                int imgX, imgY;
                                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                                    imgX = (int)((double)pt.x / dstW * g_imageDib.width);
                                    imgY = (int)((double)pt.y / dstH * g_imageDib.height);
                                    imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                                    imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
                                    
                                    POINT screenPt = {pt.x, pt.y};
                                    ClientToScreen(g_imageViewHwnd, &screenPt);
                                    updateMagnifierWindow(imgX, imgY, screenPt.x, screenPt.y);
                                }
                            }
                        }
                    }
                    updateColorPickerLabels();  // Update button text and UI
                    return 0;
                }
                if (id == IDC_RAIL_COLOR_PICKER) {
                    // Activate/deactivate rail color picker mode.
                    const ColorPickerTarget requested = ColorPickerTarget::Rail;

                    if (g_colorPickerTarget == requested) {
                        // Deactivate
                        g_colorPickerTarget = ColorPickerTarget::None;
                        if (g_magnifierHwnd) ShowWindow(g_magnifierHwnd, SW_HIDE);
                    } else {
                        // Switch to requested picker
                        g_colorPickerTarget = requested;

                        // Enable mouse tracking in ImageView to get mouse move events
                        if (g_imageViewHwnd) {
                            TRACKMOUSEEVENT tme = {};
                            tme.cbSize = sizeof(TRACKMOUSEEVENT);
                            tme.dwFlags = TME_HOVER | TME_LEAVE;
                            tme.hwndTrack = g_imageViewHwnd;
                            tme.dwHoverTime = 1;  // Immediate hover
                            TrackMouseEvent(&tme);

                            // Get current mouse position and show magnifier immediately
                            POINT pt;
                            GetCursorPos(&pt);
                            ScreenToClient(g_imageViewHwnd, &pt);

                            // Convert to image coordinates and show magnifier
                            if (!g_lastSourceFrame.empty()) {
                                RECT rc;
                                GetClientRect(g_imageViewHwnd, &rc);
                                const int dstW = std::max(0L, rc.right - rc.left);
                                const int dstH = std::max(0L, rc.bottom - rc.top);

                                int imgX, imgY;
                                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                                    imgX = (int)((double)pt.x / dstW * g_imageDib.width);
                                    imgY = (int)((double)pt.y / dstH * g_imageDib.height);
                                    imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                                    imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);

                                    POINT screenPt = {pt.x, pt.y};
                                    ClientToScreen(g_imageViewHwnd, &screenPt);
                                    updateMagnifierWindow(imgX, imgY, screenPt.x, screenPt.y);
                                }
                            }
                        }
                    }
                    updateColorPickerLabels();  // Update button text and UI
                    return 0;
                }
                if (id == kIdDiamondsColor) {
                    cv::Scalar newColor = uiControls.diamondParams.color;
                    if (chooseColor(hwnd, uiControls.diamondParams.color, newColor)) {
                        uiControls.diamondParams.color = newColor;
                        saveSettingsToDisk();  // Save settings immediately when overlay color changes
                    }
                    return 0;
                }
                if (id == kIdFeltColor) {
                    cv::Scalar newColor = uiControls.feltParams.color;
                    if (chooseColor(hwnd, uiControls.feltParams.color, newColor)) {
                        uiControls.feltParams.color = newColor;
                        saveSettingsToDisk();  // Save settings immediately when overlay color changes
                    }
                    return 0;
                }
                if (id == kIdRailsColor) {
                    cv::Scalar newColor = uiControls.railParams.color;
                    if (chooseColor(hwnd, uiControls.railParams.color, newColor)) {
                        uiControls.railParams.color = newColor;
                        saveSettingsToDisk();  // Save settings immediately when overlay color changes
                    }
                    return 0;
                }
            }

            // (Accordion toggles removed)


            // Display sidebar combobox (ID 40001)
            const int kIdDisplaySourceCombo = 40001;
            if (id == kIdDisplaySourceCombo && hwndCtl) {
                // Only handle selection changes; ignore dropdown/open/close notifications to avoid rebuilds during UI.
                if (code == CBN_SELCHANGE) {
                    const int sel = (int)SendMessage(hwndCtl, CB_GETCURSEL, 0, 0);
                    if (sel == CB_ERR) return 0;
                    
                    // Robust rule:
                    // - Index 0 is always "Test Image" (even if item-data is missing/incorrect).
                    // This avoids ambiguous cases like camera 0 vs "Test Image" if CB_SETITEMDATA fails.
                    if (sel == 0) {
                        uiControls.selectedSource = kSourceTestImage;
                        g_requestedCaptureSource.store(kSourceTestImage, std::memory_order_relaxed);
                    } else {
                        const LRESULT camData = SendMessage(hwndCtl, CB_GETITEMDATA, sel, 0);
                        if (camData == CB_ERR) {
                            // Fallback to Test Image if item-data is unavailable.
                            uiControls.selectedSource = kSourceTestImage;
                            g_requestedCaptureSource.store(kSourceTestImage, std::memory_order_relaxed);
                            return 0;
                        }
                        const int source = (int)camData; // -2 = test video, >=0 = camera index
                        uiControls.selectedSource = source;
                        g_requestedCaptureSource.store(source, std::memory_order_relaxed);
                    }

                    // Avoid briefly showing a stale frame from the previous source after switching.
                    {
                        std::lock_guard<std::mutex> lock(g_latestFrameMutex);
                        g_latestCaptureFrame.release();
                    }
                    
                    // Save settings immediately when source selection changes
                    saveSettingsToDisk();
                }
                return 0;
            }

            // Refresh camera list button (IDC_BUTTON_BASE + 280 = 30280)
            if (code == BN_CLICKED && id == 30280) {
                // Hint to capture thread to release camera before probing devices.
                g_requestedCaptureSource.store(kSourceTestImage, std::memory_order_relaxed);
                uiControls.selectedSource = kSourceTestImage;
                uiControls.currentOpenedCamera = -1;
                {
                    std::lock_guard<std::mutex> lock(g_latestFrameMutex);
                    g_latestCaptureFrame.release();
                }

                g_availableCameras = enumerateCameras();

                // If current selection no longer exists, fall back to Test Image.
                if (uiControls.selectedSource >= 0) {
                    bool found = false;
                    for (int cam : g_availableCameras) {
                        if (cam == uiControls.selectedSource) { found = true; break; }
                    }
                    if (!found) {
                        uiControls.selectedSource = kSourceTestImage;
                    }
                }

                // Save settings immediately if source selection changed
                saveSettingsToDisk();

                // Rebuild sidebar (Display page shows the refreshed list)
                destroySidebarControls();
                createSidebarControls(hwnd);
                return 0;
            }

            // If this WM_COMMAND came from a control we don't handle, stop here.
            if (hwndCtl != NULL) {
                return 0;
            }

            // Menu command (lParam == NULL)
            handleMenuCommand(id);
            layoutChildren(hwnd);
            if (uiControls.showSidebar) {
                destroySidebarControls();
                createSidebarControls(hwnd);
            }
            return 0;
        }
        case WM_HSCROLL: {
            HWND hTrackbar = (HWND)lParam;
            const int trackbarId = GetDlgCtrlID(hTrackbar);
            const int value = (int)SendMessage(hTrackbar, TBM_GETPOS, 0, 0);
            handleTrackbarChange(trackbarId, value);
            return 0;
        }
        case WM_DESTROY: {
            // Persist the current parameters before shutting down (best effort).
            saveSettingsToDisk();

            // Cleanup magnifier window
            if (g_magnifierHwnd) {
                DestroyWindow(g_magnifierHwnd);
                g_magnifierHwnd = NULL;
            }
            if (g_magnifierBitmap) {
                DeleteObject(g_magnifierBitmap);
                g_magnifierBitmap = NULL;
            }
            if (g_magnifierDC) {
                DeleteDC(g_magnifierDC);
                g_magnifierDC = NULL;
            }
            if (g_diamondColorPickerSwatchBrush) {
                DeleteObject(g_diamondColorPickerSwatchBrush);
                g_diamondColorPickerSwatchBrush = NULL;
            }
            if (g_feltColorPickerSwatchBrush) {
                DeleteObject(g_feltColorPickerSwatchBrush);
                g_feltColorPickerSwatchBrush = NULL;
            }
            KillTimer(hwnd, 1);
            stopCaptureThread();
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static int runWin32HostedApp(int argc, char** argv) {
    g_availableCameras = enumerateCameras();

    // Load previous session settings (best-effort).
    // This must happen before we start the capture thread so the initial source selection is correct.
    loadSettingsFromDisk();

    std::string imagePath = "testImage.jpg";
    if (argc > 1) imagePath = argv[1];

    g_testImage = cv::imread(imagePath);
    if (g_testImage.empty()) {
        std::string altPath = "../../" + imagePath;
        g_testImage = cv::imread(altPath);
    }
    if (g_testImage.empty()) {
        return -1;
    }

    // Resolve test video path similarly to the test image (common when running from `build/`).
    // NOTE: We don't *open* the video here; the capture thread opens it on demand when selected.
    {
        const std::string videoPath = "testVideo.mp4";
        if (std::filesystem::exists(videoPath)) {
            g_testVideoPath = videoPath;
        } else if (std::filesystem::exists("../../" + videoPath)) {
            g_testVideoPath = "../../" + videoPath;
        } else {
            // Keep default "testVideo.mp4"; if it doesn't exist, selection will simply fall back to Test Image.
            g_testVideoPath = videoPath;
        }
    }

    // Start capture thread with current selection (default is Test Video).
    g_requestedCaptureSource.store(uiControls.selectedSource, std::memory_order_relaxed);
    startCaptureThread();

    HINSTANCE hInst = GetModuleHandleW(NULL);

    // Register ImageView class
    WNDCLASSEXW wcImg{};
    wcImg.cbSize = sizeof(wcImg);
    wcImg.lpfnWndProc = ImageViewProc;
    wcImg.hInstance = hInst;
    wcImg.lpszClassName = kImageViewClass;
    wcImg.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassExW(&wcImg);

    // Register SidebarPanel class (forward WM_COMMAND / WM_HSCROLL to main window)
    WNDCLASSEXW wcSide{};
    wcSide.cbSize = sizeof(wcSide);
    wcSide.lpfnWndProc = SidebarPanelProc;
    wcSide.hInstance = hInst;
    wcSide.lpszClassName = kSidebarPanelClass;
    wcSide.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcSide.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassExW(&wcSide);

    // Register MainWindow class
    WNDCLASSEXW wcMain{};
    wcMain.cbSize = sizeof(wcMain);
    wcMain.lpfnWndProc = MainWindowProc;
    wcMain.hInstance = hInst;
    wcMain.lpszClassName = kMainWindowClass;
    wcMain.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcMain.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassExW(&wcMain);

    HWND hwnd = CreateWindowExW(
        0,
        kMainWindowClass,
        L"Billiards Trainer",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        1280, 720,
        NULL, NULL,
        hInst,
        NULL
    );

    if (!hwnd) {
        stopCaptureThread();
        return -1;
    }

    MSG msg{};
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}
#endif // _WIN32

// Menu command IDs
#define IDM_NAV_DEBUG 900
#define IDM_NAV_DISPLAY 901
#define IDM_EXPORT_CAPTURES 902
#define IDM_DEBUG_OVERLAY 1000
#define IDM_DEBUG_OVERLAY_DIAMONDS 1001
#define IDM_DEBUG_OVERLAY_FELT 1002
#define IDM_DEBUG_OVERLAY_RAIL 1003
#define IDM_DEBUG_SIDEBAR 1004
#define IDM_DISPLAY_TESTIMAGE 2001
#define IDM_DISPLAY_CAMERA_BASE 3000

// Sidebar constants
const int SIDEBAR_WIDTH = 300;
const int SIDEBAR_COLLAPSED_WIDTH = 30; // Width when collapsed (just for collapse button)
#define IDC_SIDEBAR_COLLAPSE (IDC_BUTTON_BASE + 100)
#define IDC_SIDEBAR_DIAMONDS (IDC_BUTTON_BASE + 101)
#define IDC_SIDEBAR_FELT (IDC_BUTTON_BASE + 102)
#define IDC_SIDEBAR_RAILS (IDC_BUTTON_BASE + 103)
#define IDC_SIDEBAR_RECTIFY (IDC_BUTTON_BASE + 104)

// Control IDs for sidebar controls
#define IDC_STATIC_BASE 10000
#define IDC_TRACKBAR_BASE 20000
#define IDC_BUTTON_BASE 30000
// IDC_COMBO_BASE is defined earlier in the file

// Trackbar IDs
#define IDC_DIAMOND_THRESH1 (IDC_TRACKBAR_BASE + 1)  // Maps to min_threshold
#define IDC_DIAMOND_THRESH2 (IDC_TRACKBAR_BASE + 2)  // Legacy, unused
#define IDC_DIAMOND_MINAREA (IDC_TRACKBAR_BASE + 3)
#define IDC_DIAMOND_MAXAREA (IDC_TRACKBAR_BASE + 4)
#define IDC_DIAMOND_CIRCULARITY (IDC_TRACKBAR_BASE + 5)  // 0-100 (scaled from 0.0-1.0)
#define IDC_DIAMOND_MORPH_KERNEL (IDC_TRACKBAR_BASE + 6)  // Morphological kernel size (5-31, must be odd)
#define IDC_DIAMOND_SKIP_MORPH 30250  // Skip morphological enhancement checkbox (matches kIdDiamondSkipMorph above)
#define IDC_DIAMOND_ADAPTIVE_C (IDC_TRACKBAR_BASE + 7)  // Adaptive threshold C constant
// Color picker sensitivity (tolerance) slider: 0..100 (0=strict, 100=loose)
#define IDC_DIAMOND_COLOR_SENSITIVITY (IDC_TRACKBAR_BASE + 8)
// Felt color picker sensitivity (tolerance) slider: 0..100 (0=strict, 100=loose)
#define IDC_FELT_COLOR_SENSITIVITY (IDC_TRACKBAR_BASE + 9)
// Rail color picker sensitivity (tolerance) slider: 0..100 (0=strict, 100=loose)
#define IDC_RAIL_COLOR_SENSITIVITY (IDC_TRACKBAR_BASE + 10)
// Button IDs
#define IDC_DIAMOND_COLOR (IDC_BUTTON_BASE + 1)
#define IDC_FELT_COLOR (IDC_BUTTON_BASE + 2)

// Sidebar checkbox IDs (Debug page)
#define IDC_DEBUG_OVERLAY_MASTER (IDC_BUTTON_BASE + 200)
#define IDC_DEBUG_OVERLAY_DIAMONDS_CB (IDC_BUTTON_BASE + 201)
#define IDC_DEBUG_OVERLAY_FELT_CB (IDC_BUTTON_BASE + 202)
#define IDC_DEBUG_OVERLAY_RAILS_CB (IDC_BUTTON_BASE + 203)
// Overlay style controls
#define IDC_DIAMONDS_STYLE_COLOR (IDC_BUTTON_BASE + 220)
#define IDC_DIAMONDS_STYLE_FILLED (IDC_BUTTON_BASE + 221)
#define IDC_FELT_STYLE_COLOR (IDC_BUTTON_BASE + 230)
#define IDC_FELT_STYLE_FILLED (IDC_BUTTON_BASE + 231)
#define IDC_RAILS_STYLE_COLOR (IDC_BUTTON_BASE + 240)
#define IDC_RAILS_STYLE_FILLED (IDC_BUTTON_BASE + 241)

// Sidebar combobox IDs (Display page)
#define IDC_DISPLAY_SOURCE_COMBO (IDC_COMBO_BASE + 1)
#define IDC_DISPLAY_REFRESH_CAMERAS (IDC_BUTTON_BASE + 280)

// Diamond detection info display
#define IDC_DIAMOND_COUNT (IDC_STATIC_BASE + 800)

// Overlay style trackbars
#define IDC_DIAMONDS_RADIUS (IDC_TRACKBAR_BASE + 60)
#define IDC_DIAMONDS_THICKNESS (IDC_TRACKBAR_BASE + 61)
#define IDC_DIAMONDS_ALPHA (IDC_TRACKBAR_BASE + 62)
#define IDC_FELT_ALPHA (IDC_TRACKBAR_BASE + 70)
#define IDC_FELT_THICKNESS (IDC_TRACKBAR_BASE + 71)
#define IDC_RAILS_ALPHA (IDC_TRACKBAR_BASE + 80)
#define IDC_RAILS_THICKNESS (IDC_TRACKBAR_BASE + 81)
#define IDC_RECTIFY_MARGIN_SCALE (IDC_TRACKBAR_BASE + 90)
#define IDC_RECTIFY_PAD_PX (IDC_TRACKBAR_BASE + 91)

// Global tuning trackbars
#define IDC_SMOOTHING (IDC_TRACKBAR_BASE + 95)

// Legacy HighGUI path (kept around for reference/fallback). We do NOT use this on Windows anymore.
int legacyHighGuiMain(int argc, char** argv) {
    // Enumerate available cameras
    g_availableCameras = enumerateCameras();

    // Load previous session settings (best-effort).
    loadSettingsFromDisk();
    
    // Load the test image
    std::string imagePath = "testImage.jpg";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    // Try to find the image in current directory or project root
    cv::Mat testImage = cv::imread(imagePath);
    if (testImage.empty()) {
        // Try from project root (two levels up from build/Release)
        std::string altPath = "../../" + imagePath;
        testImage = cv::imread(altPath);
        if (!testImage.empty()) {
            imagePath = altPath;
        }
    }
    
    if (testImage.empty()) {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        std::cerr << "Tried: " << imagePath << " and ../../" << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Loaded image: " << imagePath << std::endl;
    std::cout << "Image size: " << testImage.cols << "x" << testImage.rows << std::endl;
    
    // Create window (resizable)
    const std::string windowName = "Billiards Trainer - Table Detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName, onMouse, NULL);
    
    // Get the window handle and create native menu
#ifdef _WIN32
    // Wait a moment for window to be created, then get handle
    cv::waitKey(100);
    HWND cvHwnd = FindWindowA(NULL, "Billiards Trainer - Table Detection");
    if (cvHwnd == NULL) {
        // Try to find by class name (OpenCV window class)
        cvHwnd = FindWindowA("HighGUI class", NULL);
    }
    if (cvHwnd != NULL) {
        g_hwnd = cvHwnd;
        createNativeMenu(cvHwnd, g_availableCameras);
        
        // IMPORTANT:
        // OpenCV/HighGUI repaints its entire client area after each `imshow`, which can visually erase
        // child windows unless the parent window is configured to clip children.
        //
        // Symptom without this: sidebar controls briefly appear during resize (when painting pauses),
        // then disappear immediately when you stop resizing (HighGUI repaints over them).
        //
        // Fix: force the HighGUI top-level window to respect child windows during repaint.
        LONG_PTR style = GetWindowLongPtr(cvHwnd, GWL_STYLE);
        style |= WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
        SetWindowLongPtr(cvHwnd, GWL_STYLE, style);
        SetWindowPos(cvHwnd, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

        // Subclass the window to handle menu commands
        g_oldWndProc = (WNDPROC)GetWindowLongPtr(cvHwnd, GWLP_WNDPROC);
        SetWindowLongPtr(cvHwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(WindowProc));

        // Create layout containers + sidebar immediately (sidebar is enabled by default).
        updateLayout(cvHwnd);
        if (uiControls.showSidebar) {
            createSidebarControls(cvHwnd);
        }
    }
#endif
    
    // Set initial window size (fit to typical monitor, maintain aspect ratio)
    int initialWidth = 1280;
    int initialHeight = 720;
    double imageAspect = static_cast<double>(testImage.cols) / testImage.rows;
    double windowAspect = static_cast<double>(initialWidth) / (initialHeight - 30); // Account for menu bar
    
    if (imageAspect > windowAspect) {
        // Image is wider - fit to width
        initialHeight = static_cast<int>(initialWidth / imageAspect) + 30;
    } else {
        // Image is taller - fit to height
        initialWidth = static_cast<int>((initialHeight - 30) * imageAspect);
    }
    
    // Resize window first
    cv::resizeWindow(windowName, initialWidth, initialHeight);
    
    // Move window to a visible position
#ifdef _WIN32
    // Get primary monitor dimensions on Windows
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int windowX = (screenWidth - initialWidth) / 2;
    int windowY = (screenHeight - initialHeight) / 2;
    // Ensure window is on-screen (at least partially visible)
    if (windowX < 0) windowX = 50;
    if (windowY < 0) windowY = 50;
    cv::moveWindow(windowName, windowX, windowY);
#else
    // For non-Windows, use a safe default position
    cv::moveWindow(windowName, 100, 100);
#endif
    
    // Main loop
    bool running = true;
    cv::VideoCapture camera;
    bool sidebarInitialized = false;
    
    while (running) {
        // Initialize layout and sidebar on first frame (after window is fully ready)
        if (!sidebarInitialized && g_hwnd != NULL) {
            RECT clientRect;
            GetClientRect(g_hwnd, &clientRect);
            if (clientRect.right > 0 && clientRect.bottom > 0) {
                // Create layout containers first
                updateLayout(g_hwnd);
                // Then create sidebar controls if sidebar is enabled
                if (uiControls.showSidebar) {
                    destroySidebarControls();
                    createSidebarControls(g_hwnd);
                }
                sidebarInitialized = true;
            }
        }
        // Process Windows messages
#ifdef _WIN32
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT || msg.message == WM_CLOSE) {
                running = false;
            }
        }
        
        // Check if window still exists
        if (g_hwnd != NULL && !IsWindow(g_hwnd)) {
            running = false;
        }
#endif
        
        // Get current window size
        cv::Size windowSize = cv::getWindowImageRect(windowName).size();
        int windowWidth = windowSize.width;
        int windowHeight = windowSize.height;
        
        // Ensure minimum window size
        if (windowWidth < 400) windowWidth = 400;
        if (windowHeight < 330) windowHeight = 330;
        
        // Get current frame (test image, test video, or camera)
        cv::Mat currentFrame;
        if (uiControls.selectedSource == kSourceTestImage) {
            // Close capture if we're switching to test image.
            if (camera.isOpened()) {
                camera.release();
                uiControls.currentOpenedCamera = -1;
            }
            testImage.copyTo(currentFrame);
        } else if (uiControls.selectedSource == kSourceTestVideo) {
            // Test video (looping).
            //
            // NOTE: This legacy (OpenCV window) code path keeps its own capture object (not the background thread
            // used by the Win32-hosted UI path). We still implement looping for consistent behavior.
            if (!camera.isOpened() || uiControls.currentOpenedCamera != kSourceTestVideo) {
                if (camera.isOpened()) camera.release();
                uiControls.currentOpenedCamera = -1;

                camera.open(g_testVideoPath);
                if (!camera.isOpened()) camera.open("testVideo.mp4");
                if (!camera.isOpened()) camera.open("../../testVideo.mp4");
                if (camera.isOpened()) {
                    uiControls.currentOpenedCamera = kSourceTestVideo;
                } else {
                    // Fall back to test image if we can't open the video.
                    uiControls.selectedSource = kSourceTestImage;
                    testImage.copyTo(currentFrame);
                }
            }

            if (camera.isOpened()) {
                camera >> currentFrame;
                if (currentFrame.empty()) {
                    // EOF -> rewind and continue.
                    camera.set(cv::CAP_PROP_POS_FRAMES, 0);
                    camera >> currentFrame;
                }
                // For camera input, no frame index
                g_lastFrameIndex = -1;
            }
            if (currentFrame.empty()) {
                testImage.copyTo(currentFrame);
                g_lastFrameIndex = -1; // Image input, no frame index
            }
        } else {
            // Camera index (>= 0)
            const int requestedCamera = uiControls.selectedSource;

            // If camera is open but different camera selected, close and reopen.
            if (camera.isOpened() && uiControls.currentOpenedCamera != requestedCamera) {
                camera.release();
                uiControls.currentOpenedCamera = -1;
            }

            // Open camera if not already open.
            if (!camera.isOpened()) {
                camera.open(requestedCamera);
                if (camera.isOpened()) {
                    uiControls.currentOpenedCamera = requestedCamera;
                    // Set some camera properties for better performance
                    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                } else {
                    std::cerr << "Warning: Could not open camera " << requestedCamera << std::endl;
                    uiControls.selectedSource = kSourceTestImage; // Fall back to test image
                    testImage.copyTo(currentFrame);
                }
            }

            if (camera.isOpened()) {
                camera >> currentFrame;
            }
            if (currentFrame.empty()) {
                testImage.copyTo(currentFrame); // Fallback if camera fails
            }
        }
        
        // Process the image
        cv::Mat processedImage;
        currentFrame.copyTo(processedImage);
        
        // Apply overlays only if master overlay toggle is enabled
        // Master toggle acts as a gate - individual toggles control what's shown
        if (uiControls.showOverlay) {
            // Apply felt detection overlay if enabled (using felt_detection module)
            if (uiControls.showFelt) {
                FeltDetectionResult feltResult = detectFelt(currentFrame, uiControls.feltParams);
                drawFeltOverlay(processedImage, feltResult, uiControls.feltParams);
            }
        }
        
        // Keep a copy of the last overlaid frame for menu-driven capture export.
        g_lastProcessedFrame = processedImage.clone();
        // Keep source frame for color picking
        g_lastSourceFrame = currentFrame.clone();
        
        // Store scaled image position for coordinate conversion
        int sidebarWidth = (uiControls.showSidebar && !uiControls.sidebarCollapsed) ? SIDEBAR_WIDTH : 
                          (uiControls.showSidebar && uiControls.sidebarCollapsed) ? SIDEBAR_COLLAPSED_WIDTH : 0;
        int availableWidth = windowWidth - sidebarWidth;
        int availableHeight = windowHeight - 30;
        double frameAspect = static_cast<double>(processedImage.cols) / processedImage.rows;
        double availableAspect = static_cast<double>(availableWidth) / availableHeight;
        
        int scaledWidth, scaledHeight;
        if (frameAspect > availableAspect) {
            scaledWidth = availableWidth;
            scaledHeight = static_cast<int>(availableWidth / frameAspect);
        } else {
            scaledHeight = availableHeight;
            scaledWidth = static_cast<int>(availableHeight * frameAspect);
        }
        g_scaledImageX = (availableWidth - scaledWidth) / 2;
        g_scaledImageY = 30 + (availableHeight - scaledHeight) / 2;
        g_scaledImageW = scaledWidth;
        g_scaledImageH = scaledHeight;
        
        // Scale the processed image
        cv::Mat scaledImage;
        cv::resize(processedImage, scaledImage, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_LINEAR);
        
        // Create display image
        cv::Mat displayImage = cv::Mat::zeros(windowHeight, windowWidth, CV_8UC3);
        displayImage.setTo(cv::Scalar(30, 30, 30)); // Dark background
        
        // Center the scaled image in the available space
        int imageX = (availableWidth - scaledWidth) / 2;
        int imageY = 30 + (availableHeight - scaledHeight) / 2; // Start below menu bar
        
        // Copy scaled image to display
        cv::Rect imageROI(imageX, imageY, scaledWidth, scaledHeight);
        scaledImage.copyTo(displayImage(imageROI));
        
        // Sidebar controls are managed separately via Windows API
        // No need to draw here - controls are native Windows elements
        
        // Display image
        cv::imshow(windowName, displayImage);
        
        // Handle keyboard input
        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC
            running = false;
        }
        else if (key == 's' || key == 'S') {
            std::filesystem::path outDir;
            std::string err;
            (void)exportCapturesToDisk(processedImage, &outDir, &err);
        }
    }
    
    // Cleanup: Release all resources
    if (camera.isOpened()) {
        camera.release();
    }
    
    // Destroy all OpenCV windows
    cv::destroyAllWindows();
    
    // Restore original window procedure before closing
#ifdef _WIN32
    if (g_hwnd != NULL && g_oldWndProc != NULL) {
        SetWindowLongPtr(g_hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_oldWndProc));
    }
    
    // Destroy menu if it exists
    if (g_hwnd != NULL) {
        HMENU hMenu = GetMenu(g_hwnd);
        if (hMenu != NULL) {
            DestroyMenu(hMenu);
            SetMenu(g_hwnd, NULL);
        }
    }
#endif
    
    return 0;
}

// Create native Windows menu bar
void createNativeMenu(HWND hwnd, const std::vector<int>& cameras) {
    HMENU hMenuBar = CreateMenu();
    
    // Top-level menu items are now direct commands (no dropdown).
    // Clicking "Debug" or "Display" switches the sidebar page.
    (void)cameras;
    AppendMenuW(hMenuBar, MF_STRING | ((uiControls.sidebarPage == SidebarPage::Debug) ? MF_CHECKED : MF_UNCHECKED),
                IDM_NAV_DEBUG, L"Debug");
    AppendMenuW(hMenuBar, MF_STRING | ((uiControls.sidebarPage == SidebarPage::Display) ? MF_CHECKED : MF_UNCHECKED),
                IDM_NAV_DISPLAY, L"Display");
    AppendMenuW(hMenuBar, MF_STRING, IDM_EXPORT_CAPTURES, L"Export Captures");
    
    // Set the menu bar
    SetMenu(hwnd, hMenuBar);
    DrawMenuBar(hwnd);
}

// Update layout: position image container and sidebar panel (like a flexbox)
void updateLayout(HWND hwnd) {
    if (!hwnd || !IsWindow(hwnd)) return;
    
    // Get window client area
    RECT clientRect;
    GetClientRect(hwnd, &clientRect);
    int windowWidth = clientRect.right;
    int windowHeight = clientRect.bottom;
    
    if (windowWidth == 0 || windowHeight == 0) {
        return;
    }
    
    int sidebarWidth = (uiControls.showSidebar && !uiControls.sidebarCollapsed) ? SIDEBAR_WIDTH : 
                      (uiControls.showSidebar && uiControls.sidebarCollapsed) ? SIDEBAR_COLLAPSED_WIDTH : 0;
    int sidebarY = 30; // Below menu bar
    int sidebarHeight = windowHeight - 30;
    int sidebarX = windowWidth - sidebarWidth;
    
    // Update or create image container (left side)
    int imageWidth = windowWidth - sidebarWidth;
    int imageHeight = windowHeight - 30;
    
    if (g_imageContainer == NULL || !IsWindow(g_imageContainer)) {
        // Create image container window
        g_imageContainer = CreateWindowW(
            L"STATIC",
            L"",
            WS_VISIBLE | WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
            0, sidebarY, imageWidth, imageHeight,
            hwnd,
            (HMENU)(INT_PTR)(IDC_STATIC_BASE + 300),
            NULL, NULL
        );
        
        // Set black background for image area
        HBRUSH hBrush = CreateSolidBrush(RGB(0, 0, 0));
        SetClassLongPtr(g_imageContainer, GCLP_HBRBACKGROUND, (LONG_PTR)hBrush);
    } else {
        // Update image container position and size
        SetWindowPos(g_imageContainer, HWND_BOTTOM, 0, sidebarY, imageWidth, imageHeight,
                     SWP_SHOWWINDOW | SWP_NOZORDER);
    }
    
    // Update or create sidebar panel (right side)
    if (uiControls.showSidebar) {
        // Defensive: ensure a visible width when enabled.
        if (sidebarWidth <= 0) {
            sidebarWidth = uiControls.sidebarCollapsed ? SIDEBAR_COLLAPSED_WIDTH : SIDEBAR_WIDTH;
            sidebarX = windowWidth - sidebarWidth;
        }
        if (g_sidebarPanel == NULL || !IsWindow(g_sidebarPanel)) {
            // Create the sidebar panel as a child window
            g_sidebarPanel = CreateWindowW(
                L"STATIC",
                L"",
                WS_VISIBLE | WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_BORDER,
                sidebarX, sidebarY, sidebarWidth, sidebarHeight,
                hwnd,
                (HMENU)(INT_PTR)(IDC_STATIC_BASE + 200),
                NULL, NULL
            );
            
            // Set background color to match Windows panel
            HBRUSH hBrush = CreateSolidBrush(RGB(240, 240, 240));
            SetClassLongPtr(g_sidebarPanel, GCLP_HBRBACKGROUND, (LONG_PTR)hBrush);
            
            // Force panel to be visible and on top
            ShowWindow(g_sidebarPanel, SW_SHOW);
            SetWindowPos(g_sidebarPanel, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
            UpdateWindow(g_sidebarPanel);
        } else {
            // Update position and size of existing panel
            SetWindowPos(g_sidebarPanel, HWND_TOP, sidebarX, sidebarY, sidebarWidth, sidebarHeight,
                         SWP_SHOWWINDOW | SWP_NOZORDER);
            ShowWindow(g_sidebarPanel, SW_SHOW);
            UpdateWindow(g_sidebarPanel);
        }
    } else {
        // Hide sidebar if disabled
        if (g_sidebarPanel && IsWindow(g_sidebarPanel)) {
            ShowWindow(g_sidebarPanel, SW_HIDE);
        }
    }
}

// Create native Windows controls for sidebar with context switching
void createSidebarControls(HWND hwnd) {
    // NOTE:
    // In the Win32-hosted UI, `g_sidebarPanel` is created/owned by the Win32 main window
    // and is *side-by-side* with the image view. OpenCV/HighGUI should not be involved here.
    (void)hwnd;

#ifdef _WIN32
    if (!g_sidebarPanel || !IsWindow(g_sidebarPanel)) {
        std::cerr << "[ERROR] createSidebarControls: g_sidebarPanel is invalid\n";
        return;
    }
#else
    return;
#endif
    
    // Destroy existing controls in the panel (keep scroll position)
    EnumChildWindows(g_sidebarPanel, [](HWND hwnd, LPARAM) -> BOOL {
        DestroyWindow(hwnd);
        return TRUE;
    }, 0);
    
    if (uiControls.sidebarCollapsed) {
        // Just show collapse button to expand
        HWND hBtn = CreateWindowW(L"BUTTON", L">", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                     5, 5, 20, 25, g_sidebarPanel, (HMENU)(INT_PTR)IDC_SIDEBAR_COLLAPSE, NULL, NULL);
        if (hBtn) {
            ShowWindow(hBtn, SW_SHOW);
            UpdateWindow(hBtn);
        }
        return;
    }
    
    // Get panel dimensions for positioning
    RECT panelRect;
    GetClientRect(g_sidebarPanel, &panelRect);
    int panelWidth = panelRect.right;
    int panelHeight = panelRect.bottom;
    
    // Padding/margins inside the sidebar so content doesn't hug the edges.
    const int padL = 16;
    const int padR = 16;
    const int padTop = 12;
    const int gap = 12;
    const int valueWidth = 52;
    const int dividerPadY = 6;

    int yPos = padTop;
    int lineHeight = 30;
    int labelWidth = 120;
    int trackbarHeight = 20;
    int xPos = padL;
    int buttonWidth = 80;
    int buttonHeight = 25;
    int usableWidth = std::max(120, panelWidth - padL - padR);

    auto addDivider = [&](int y) {
        HWND h = CreateWindowW(L"STATIC", L"", WS_VISIBLE | WS_CHILD | SS_ETCHEDHORZ,
                               xPos, y - g_sidebarScrollPos, usableWidth, 1, g_sidebarPanel, NULL, NULL, NULL);
        applyFont(h, false);
        return h;
    };

    // Section header helper
    auto addHeader = [&](const wchar_t* title) {
        HWND hTitle = CreateWindowW(L"STATIC", title, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                    xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel, NULL, NULL, NULL);
        applyFont(hTitle, true);
        yPos += lineHeight;
        addDivider(yPos - dividerPadY);
        yPos += dividerPadY;
    };
    
    // Accordion header helper (clickable button with expand/collapse indicator)
    auto addAccordionHeader = [&](const wchar_t* title, int buttonId, bool& expanded) {
        const wchar_t* indicator = expanded ? L"\u25BC " : L"\u25BA "; // ▼ for expanded, ▶ for collapsed
        std::wstring buttonText = std::wstring(indicator) + title;
        HWND hHeader = CreateWindowW(L"BUTTON", buttonText.c_str(), WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON | BS_LEFT,
                                    xPos, yPos - g_sidebarScrollPos, usableWidth, 24, g_sidebarPanel,
                                    (HMENU)(INT_PTR)buttonId, NULL, NULL);
        if (hHeader) {
            applyFont(hHeader, true);
        }
        yPos += lineHeight + 4;
    };

    // Helper to create a static label - relative to panel
    auto createLabel = [&](const wchar_t* text, int y, int id) {
        HWND h = CreateWindowW(L"STATIC", text, WS_VISIBLE | WS_CHILD | SS_LEFT,
                           xPos, y - g_sidebarScrollPos, labelWidth, 20, g_sidebarPanel, (HMENU)(INT_PTR)id, NULL, NULL);
        applyFont(h, false);
        return h;
    };

    // Helper to create a value display - relative to panel (right-aligned to a standard trackbar layout)
    auto createValueLabel = [&](int y, int id) {
        wchar_t buffer[32]{};
        const int trackbarWidth = std::max(80, usableWidth - labelWidth - valueWidth - 10);
        HWND h = CreateWindowW(L"STATIC", buffer, WS_VISIBLE | WS_CHILD | SS_LEFT,
                           xPos + labelWidth + trackbarWidth + 5, y - g_sidebarScrollPos, valueWidth, 20,
                           g_sidebarPanel, (HMENU)(INT_PTR)id, NULL, NULL);
        applyFont(h, false);
        return h;
    };

    // Helper to create a trackbar - relative to panel (standard label+trackbar+value layout)
    auto createTrackbar = [&](int id, int y, int minVal, int maxVal, int currentVal) {
        // Trackbar width adapts to panel width, leaving room for the value label.
        const int trackbarWidth = std::max(80, usableWidth - labelWidth - valueWidth - 10);
        HWND hTrackbar = CreateWindowW(TRACKBAR_CLASSW, L"", 
                                      WS_VISIBLE | WS_CHILD | TBS_AUTOTICKS | TBS_HORZ,
                                      xPos + labelWidth, y - g_sidebarScrollPos, trackbarWidth, trackbarHeight,
                                      g_sidebarPanel, (HMENU)(INT_PTR)id, NULL, NULL);
        if (hTrackbar) {
            applyFont(hTrackbar, false);
            SetWindowTheme(hTrackbar, L"Explorer", NULL);
        }
        SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(minVal, maxVal));
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, currentVal);
        SendMessage(hTrackbar, TBM_SETTICFREQ, (maxVal - minVal) / 10, 0);
        return hTrackbar;
    };

    // Inline trackbar helper (used for "Alpha" slider next to the Color button)
    auto createInlineTrackbar = [&](int id, int x, int y, int width, int minVal, int maxVal, int currentVal) {
        HWND hTrackbar = CreateWindowW(TRACKBAR_CLASSW, L"",
                                       WS_VISIBLE | WS_CHILD | TBS_AUTOTICKS | TBS_HORZ,
                                       x, y - g_sidebarScrollPos, std::max(40, width), trackbarHeight,
                                       g_sidebarPanel, (HMENU)(INT_PTR)id, NULL, NULL);
        if (hTrackbar) {
            applyFont(hTrackbar, false);
            SetWindowTheme(hTrackbar, L"Explorer", NULL);
        }
        SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(minVal, maxVal));
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, currentVal);
        return hTrackbar;
    };
    
    // Collapse button (top right) - relative to panel
    HWND hCollapseBtn = CreateWindowW(L"BUTTON", L"<", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                 panelWidth - 25, 5, 20, 25, g_sidebarPanel, 
                 (HMENU)(INT_PTR)IDC_SIDEBAR_COLLAPSE, NULL, NULL);
    if (hCollapseBtn) {
        ShowWindow(hCollapseBtn, SW_SHOW);
        UpdateWindow(hCollapseBtn);
        applyFont(hCollapseBtn, true);
    }
    
    yPos += 35;

    // Title reflecting the current page
    const wchar_t* pageTitle = (uiControls.sidebarPage == SidebarPage::Debug) ? L"Debug" : L"Display";
    {
        HWND hTitle = CreateWindowW(L"STATIC", pageTitle, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                    xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel, NULL, NULL, NULL);
        applyFont(hTitle, true);
    }
    yPos += lineHeight;
    addDivider(yPos - dividerPadY);
    yPos += dividerPadY;

    if (uiControls.sidebarPage == SidebarPage::Debug) {
        // ===== Debug page sections (ordered): Global, Felt, Diamonds =====

        // GLOBAL section
        //
        // Cross-cutting overlay controls:
        // - master overlay gate
        // - overlay smoothing strength
        {
            addHeader(L"Global");

            // Master overlay toggle
            {
                HWND h = CreateWindowW(L"BUTTON", L"Overlay (master)", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                       xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel,
                                       (HMENU)(INT_PTR)IDC_DEBUG_OVERLAY_MASTER, NULL, NULL);
                applyFont(h, false);
                SendMessage(h, BM_SETCHECK, uiControls.showOverlay ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Global smoothing
            createLabel(L"Smoothing:", yPos, IDC_STATIC_BASE + 400);
            createTrackbar(IDC_SMOOTHING, yPos, 0, 100, uiControls.smoothingPercent);
            createValueLabel(yPos, IDC_STATIC_BASE + 401);
            yPos += lineHeight + gap;
        }

        // Shared layout constants for feature groups
        const int rowH = 24;
        const int enabledW = 90;
        const int filledW = 70;
        const int colorW = 72;
        const int colorH = 24;

        auto addFeatureGroup = [&](const wchar_t* title,
                                   int enabledId,
                                   bool enabledValue,
                                   int colorButtonId,
                                   int alphaTrackbarId,
                                   int alphaCurrentVal,
                                   int alphaValueLabelId,
                                   int filledId,
                                   bool filledValue,
                                   int& ioYPos) {
            addHeader(title);

            // Row 1: Enabled
            {
                HWND hEnabled = CreateWindowW(L"BUTTON", L"Enabled", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                              xPos, ioYPos - g_sidebarScrollPos, usableWidth, rowH, g_sidebarPanel,
                                              (HMENU)(INT_PTR)enabledId, NULL, NULL);
                applyFont(hEnabled, false);
                SendMessage(hEnabled, BM_SETCHECK, enabledValue ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            ioYPos += lineHeight;

            // Row 2: Color + Filled
            {
                HWND hColor = CreateWindowW(L"BUTTON", L"Color\u2026", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                            xPos, (ioYPos - 2) - g_sidebarScrollPos, colorW, colorH, g_sidebarPanel,
                                            (HMENU)(INT_PTR)colorButtonId, NULL, NULL);
                applyFont(hColor, false);

                // Right-align the "Filled" checkbox.
                const int filledX = xPos + usableWidth - filledW;
                HWND hFill = CreateWindowW(L"BUTTON", L"Filled", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                           filledX, ioYPos - g_sidebarScrollPos, filledW, rowH, g_sidebarPanel,
                                           (HMENU)(INT_PTR)filledId, NULL, NULL);
                applyFont(hFill, false);
                SendMessage(hFill, BM_SETCHECK, filledValue ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            ioYPos += lineHeight;

            // Row 3: Alpha (separate line, per request)
            {
                createLabel(L"Alpha:", ioYPos, IDC_STATIC_BASE + 450);
                createTrackbar(alphaTrackbarId, ioYPos, 0, 255, alphaCurrentVal);
                createValueLabel(ioYPos, alphaValueLabelId);
            }
            ioYPos += lineHeight;
        };

        // FELT section
        //
        // Split into:
        // - Input params: felt color + sensitivity (what pixels are considered "felt")
        // - Output params: overlay styling (how we render the detected felt contour)
        {
            addAccordionHeader(L"Felt", IDC_SIDEBAR_FELT, uiControls.feltExpanded);

            if (uiControls.feltExpanded) {
                auto createFullWidthLabel = [&](const wchar_t* text, int y, int id) {
                    HWND h = CreateWindowW(L"STATIC", text, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                           xPos, y - g_sidebarScrollPos, usableWidth, 18, g_sidebarPanel,
                                           (HMENU)(INT_PTR)id, NULL, NULL);
                    applyFont(h, false);
                    return h;
                };

            // Pick felt color (click-to-sample)
            {
                const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Felt);
                const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Felt Color";
                HWND hPicker = CreateWindowW(L"BUTTON", btnText, WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                             xPos, yPos - g_sidebarScrollPos, usableWidth, 28, g_sidebarPanel,
                                             (HMENU)(INT_PTR)IDC_FELT_COLOR_PICKER, NULL, NULL);
                applyFont(hPicker, true);
                yPos += 32;
            }

            // Swatch (tightened height)
            {
                createFullWidthLabel(L"Picked Color:", yPos, IDC_STATIC_BASE + 600);
                yPos += lineHeight;

                const cv::Vec3b bgr = uiControls.feltParams.pickedBGR;
                const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]); // BGR->RGB

                HWND hSwatch = CreateWindowW(L"STATIC", L"", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_NOTIFY,
                                             xPos + 10, yPos - g_sidebarScrollPos, usableWidth - 20, 28,
                                             g_sidebarPanel, (HMENU)(INT_PTR)IDC_FELT_COLOR_PICKER_SWATCH, NULL, NULL);
                if (hSwatch) {
                    setSwatchBrush(g_feltColorPickerSwatchBrush, swatchColor);
                    InvalidateRect(hSwatch, NULL, TRUE);
                }
                yPos += 34;
            }

            // Sensitivity slider
            {
                createLabel(L"Sensitivity:", yPos, IDC_STATIC_BASE + 601);
                createTrackbar(IDC_FELT_COLOR_SENSITIVITY, yPos, 0, 100, uiControls.feltParams.colorSensitivity);
                createValueLabel(yPos, IDC_STATIC_BASE + 602);
                yPos += lineHeight + 2;
            }

            // Readouts (full width so "Range" doesn't clip)
            {
                createFullWidthLabel(L"BGR: --", yPos, IDC_FELT_COLOR_PICKER_BGR);
                yPos += lineHeight;
                createFullWidthLabel(L"HSV: --", yPos, IDC_FELT_COLOR_PICKER_HSV);
                yPos += lineHeight;
                createFullWidthLabel(L"Range: --", yPos, IDC_FELT_COLOR_PICKER_RANGE);
                yPos += lineHeight + gap;
            }

            // Separator between felt detection parameters and overlay styling
            addDivider(yPos - 2);
            yPos += 6;

            // Show/Hide
            {
                HWND hEnabled = CreateWindowW(L"BUTTON", L"Show/Hide", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                              xPos, yPos - g_sidebarScrollPos, usableWidth, rowH, g_sidebarPanel,
                                              (HMENU)(INT_PTR)IDC_DEBUG_OVERLAY_FELT_CB, NULL, NULL);
                applyFont(hEnabled, false);
                SendMessage(hEnabled, BM_SETCHECK, uiControls.showFelt ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Style row: Color + Filled
            {
                HWND hColor = CreateWindowW(L"BUTTON", L"Color\u2026", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                            xPos, (yPos - 2) - g_sidebarScrollPos, colorW, colorH, g_sidebarPanel,
                                            (HMENU)(INT_PTR)IDC_FELT_STYLE_COLOR, NULL, NULL);
                applyFont(hColor, false);

                const int filledX = xPos + usableWidth - filledW;
                HWND hFill = CreateWindowW(L"BUTTON", L"Filled", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                           filledX, yPos - g_sidebarScrollPos, filledW, rowH, g_sidebarPanel,
                                           (HMENU)(INT_PTR)IDC_FELT_STYLE_FILLED, NULL, NULL);
                applyFont(hFill, false);
                SendMessage(hFill, BM_SETCHECK, uiControls.feltParams.isFilled ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Alpha (value label ID must remain stable: IDC_STATIC_BASE + 510)
            {
                createLabel(L"Alpha:", yPos, IDC_STATIC_BASE + 450);
                createTrackbar(IDC_FELT_ALPHA, yPos, 0, 255, uiControls.feltParams.fillAlpha);
                createValueLabel(yPos, IDC_STATIC_BASE + 510);
                yPos += lineHeight;
            }

            // Outline thickness
            {
                createLabel(L"Outline:", yPos, IDC_STATIC_BASE + 118);
                createTrackbar(IDC_FELT_THICKNESS, yPos, 1, 10, uiControls.feltParams.outlineThicknessPx);
                createValueLabel(yPos, IDC_STATIC_BASE + 119);
                yPos += lineHeight + gap;
            }
            } // End of feltExpanded block
        }

        // RAILS section
        {
            addAccordionHeader(L"Rails", IDC_SIDEBAR_RAILS, uiControls.railsExpanded);

            if (uiControls.railsExpanded) {
                // Lambda for creating full-width labels (same as felt section)
                auto createFullWidthLabel = [&](const wchar_t* text, int y, int id) {
                    HWND h = CreateWindowW(L"STATIC", text, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                           xPos, y - g_sidebarScrollPos, usableWidth, 18, g_sidebarPanel,
                                           (HMENU)(INT_PTR)id, NULL, NULL);
                    applyFont(h, false);
                    return h;
                };

                // Show/Hide
                {
                HWND hEnabled = CreateWindowW(L"BUTTON", L"Show/Hide", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                              xPos, yPos - g_sidebarScrollPos, usableWidth, rowH, g_sidebarPanel,
                                              (HMENU)(INT_PTR)IDC_DEBUG_OVERLAY_RAILS_CB, NULL, NULL);
                applyFont(hEnabled, false);
                SendMessage(hEnabled, BM_SETCHECK, uiControls.showRails ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Pick rail color (click-to-sample)
            {
                const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Rail);
                const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Rail Color";
                HWND hPicker = CreateWindowW(L"BUTTON", btnText, WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                             xPos, yPos - g_sidebarScrollPos, usableWidth, 28, g_sidebarPanel,
                                             (HMENU)(INT_PTR)IDC_RAIL_COLOR_PICKER, NULL, NULL);
                applyFont(hPicker, true);
                yPos += 32;
            }

            // Swatch
            {
                createFullWidthLabel(L"Picked Color:", yPos, IDC_STATIC_BASE + 700);
                yPos += lineHeight;

                const cv::Vec3b bgr = uiControls.railParams.pickedBGR;
                const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]); // BGR->RGB

                HWND hSwatch = CreateWindowW(L"STATIC", L"", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_NOTIFY,
                                             xPos + 10, yPos - g_sidebarScrollPos, usableWidth - 20, 28,
                                             g_sidebarPanel, (HMENU)(INT_PTR)IDC_RAIL_COLOR_PICKER_SWATCH, NULL, NULL);
                if (hSwatch) {
                    setSwatchBrush(g_railColorPickerSwatchBrush, swatchColor);
                    InvalidateRect(hSwatch, NULL, TRUE);
                }
                yPos += 34;
            }

            // Sensitivity slider
            {
                createLabel(L"Sensitivity:", yPos, IDC_STATIC_BASE + 701);
                createTrackbar(IDC_RAIL_COLOR_SENSITIVITY, yPos, 0, 100, uiControls.railParams.colorSensitivity);
                createValueLabel(yPos, IDC_STATIC_BASE + 702);
                yPos += lineHeight + 2;
            }

            // Readouts
            {
                createFullWidthLabel(L"BGR: --", yPos, IDC_RAIL_COLOR_PICKER_BGR);
                yPos += lineHeight;
                createFullWidthLabel(L"HSV: --", yPos, IDC_RAIL_COLOR_PICKER_HSV);
                yPos += lineHeight;
                createFullWidthLabel(L"Range: --", yPos, IDC_RAIL_COLOR_PICKER_RANGE);
                yPos += lineHeight + gap;
            }

            // Separator between rail detection parameters and overlay styling
            addDivider(yPos - 2);
            yPos += 6;

            // Style row: Color + Filled
            {
                HWND hColor = CreateWindowW(L"BUTTON", L"Color\u2026", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                            xPos, (yPos - 2) - g_sidebarScrollPos, colorW, colorH, g_sidebarPanel,
                                            (HMENU)(INT_PTR)IDC_RAILS_STYLE_COLOR, NULL, NULL);
                applyFont(hColor, false);

                const int filledX = xPos + usableWidth - filledW;
                HWND hFill = CreateWindowW(L"BUTTON", L"Filled", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                           filledX, yPos - g_sidebarScrollPos, filledW, rowH, g_sidebarPanel,
                                           (HMENU)(INT_PTR)IDC_RAILS_STYLE_FILLED, NULL, NULL);
                applyFont(hFill, false);
                SendMessage(hFill, BM_SETCHECK, uiControls.railParams.isFilled ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Alpha
            {
                createLabel(L"Alpha:", yPos, IDC_STATIC_BASE + 462);
                createTrackbar(IDC_RAILS_ALPHA, yPos, 0, 255, uiControls.railParams.fillAlpha);
                createValueLabel(yPos, IDC_STATIC_BASE + 522);
                yPos += lineHeight;
            }

            // Outline thickness
            {
                createLabel(L"Outline:", yPos, IDC_STATIC_BASE + 122);
                createTrackbar(IDC_RAILS_THICKNESS, yPos, 1, 10, uiControls.railParams.outlineThicknessPx);
                createValueLabel(yPos, IDC_STATIC_BASE + 123);
                yPos += lineHeight + gap;
            }
            } // End of railsExpanded block
        }

        // DIAMONDS section
        {
            addAccordionHeader(L"Diamonds", IDC_SIDEBAR_DIAMONDS, uiControls.diamondsExpanded);

            if (uiControls.diamondsExpanded) {
                // Lambda for creating full-width labels (same as felt/rail sections)
                auto createFullWidthLabel = [&](const wchar_t* text, int y, int id) {
                    HWND h = CreateWindowW(L"STATIC", text, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                           xPos, y - g_sidebarScrollPos, usableWidth, 18, g_sidebarPanel,
                                           (HMENU)(INT_PTR)id, NULL, NULL);
                    applyFont(h, false);
                    return h;
                };

                // Pick diamond color (click-to-sample)
                {
                const bool isActive = (g_colorPickerTarget == ColorPickerTarget::Diamond);
                const wchar_t* btnText = isActive ? L"Cancel (Click to Pick)" : L"Pick Diamond Color";
                HWND hPicker = CreateWindowW(L"BUTTON", btnText, WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                             xPos, yPos - g_sidebarScrollPos, usableWidth, 28, g_sidebarPanel,
                                             (HMENU)(INT_PTR)IDC_DIAMOND_COLOR_PICKER, NULL, NULL);
                applyFont(hPicker, true);
                yPos += 32;
            }

            // Swatch (tightened height)
            {
                createFullWidthLabel(L"Picked Color:", yPos, IDC_STATIC_BASE + 500);
                yPos += lineHeight;

                const cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
                const COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]);

                HWND hSwatch = CreateWindowW(L"STATIC", L"", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_NOTIFY,
                                             xPos + 10, yPos - g_sidebarScrollPos, usableWidth - 20, 28,
                                             g_sidebarPanel, (HMENU)(INT_PTR)IDC_DIAMOND_COLOR_PICKER_SWATCH, NULL, NULL);
                if (hSwatch) {
                    setSwatchBrush(g_diamondColorPickerSwatchBrush, swatchColor);
                    InvalidateRect(hSwatch, NULL, TRUE);
                }
                yPos += 34;
            }

            // Sensitivity slider
            {
                createLabel(L"Sensitivity:", yPos, IDC_STATIC_BASE + 501);
                createTrackbar(IDC_DIAMOND_COLOR_SENSITIVITY, yPos, 0, 100, uiControls.diamondParams.colorSensitivity);
                createValueLabel(yPos, IDC_STATIC_BASE + 502);
                yPos += lineHeight;
            }

            // Readouts (full width so "Range" doesn't clip)
            {
                createFullWidthLabel(L"BGR: --", yPos, IDC_DIAMOND_COLOR_PICKER_BGR);
                yPos += lineHeight;
                createFullWidthLabel(L"HSV: --", yPos, IDC_DIAMOND_COLOR_PICKER_HSV);
                yPos += lineHeight;
                createFullWidthLabel(L"Range: --", yPos, IDC_DIAMOND_COLOR_PICKER_RANGE);
                yPos += lineHeight + gap;
            }

            // Separator between diamond detection parameters and overlay styling
            addDivider(yPos - 2);
            yPos += 6;

            // Show/Hide
            {
                HWND hEnabled = CreateWindowW(L"BUTTON", L"Show/Hide", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                              xPos, yPos - g_sidebarScrollPos, usableWidth, rowH, g_sidebarPanel,
                                              (HMENU)(INT_PTR)IDC_DEBUG_OVERLAY_DIAMONDS_CB, NULL, NULL);
                applyFont(hEnabled, false);
                SendMessage(hEnabled, BM_SETCHECK, uiControls.showDiamonds ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Detection count info
            {
                wchar_t countText[64];
                // Safely access numDiamondsDetected - default to 0 if diamonds not yet detected
                const int numDiamonds = latestAnalysis.diamonds.ok ? latestAnalysis.diamonds.numDiamondsDetected : 0;
                swprintf_s(countText, L"Detected: %d / 18", numDiamonds);
                HWND hCount = CreateWindowW(L"STATIC", countText, WS_VISIBLE | WS_CHILD | SS_LEFT,
                                           xPos, yPos - g_sidebarScrollPos, usableWidth, 18, g_sidebarPanel,
                                           (HMENU)(INT_PTR)IDC_DIAMOND_COUNT, NULL, NULL);
                applyFont(hCount, false);
            }
            yPos += lineHeight;

            // Separator
            addDivider(yPos - 2);
            yPos += 6;

            // Style row: Color + Filled
            {
                HWND hColor = CreateWindowW(L"BUTTON", L"Color\u2026", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                            xPos, (yPos - 2) - g_sidebarScrollPos, colorW, colorH, g_sidebarPanel,
                                            (HMENU)(INT_PTR)IDC_DIAMONDS_STYLE_COLOR, NULL, NULL);
                applyFont(hColor, false);

                const int filledX = xPos + usableWidth - filledW;
                HWND hFill = CreateWindowW(L"BUTTON", L"Filled", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                           filledX, yPos - g_sidebarScrollPos, filledW, rowH, g_sidebarPanel,
                                           (HMENU)(INT_PTR)IDC_DIAMONDS_STYLE_FILLED, NULL, NULL);
                applyFont(hFill, false);
                SendMessage(hFill, BM_SETCHECK, uiControls.diamondParams.isFilled ? BST_CHECKED : BST_UNCHECKED, 0);
            }
            yPos += lineHeight;

            // Alpha
            {
                createLabel(L"Alpha:", yPos, IDC_STATIC_BASE + 463);
                createTrackbar(IDC_DIAMONDS_ALPHA, yPos, 0, 255, uiControls.diamondParams.fillAlpha);
                createValueLabel(yPos, IDC_STATIC_BASE + 523);
                yPos += lineHeight;
            }

            // Outline thickness
            {
                createLabel(L"Outline:", yPos, IDC_STATIC_BASE + 124);
                createTrackbar(IDC_DIAMONDS_THICKNESS, yPos, 1, 10, uiControls.diamondParams.outlineThicknessPx);
                createValueLabel(yPos, IDC_STATIC_BASE + 125);
                yPos += lineHeight + gap;
            }
            } // End of diamondsExpanded block
        }

        // RECTIFICATION section
        {
            addAccordionHeader(L"Rectification", IDC_SIDEBAR_RECTIFY, uiControls.rectifyExpanded);
            
            if (uiControls.rectifyExpanded) {
                // Margin Scale slider (1.00 to 1.40, step 0.01, default 1.18)
                // Store as integer (100-140) for trackbar, convert to float
                {
                    createLabel(L"Margin Scale:", yPos, IDC_STATIC_BASE + 470);
                    int marginScaleInt = static_cast<int>(std::round(uiControls.rectifyMarginScale * 100.0f));
                    marginScaleInt = std::clamp(marginScaleInt, 100, 140);
                    createTrackbar(IDC_RECTIFY_MARGIN_SCALE, yPos, 100, 140, marginScaleInt);
                    createValueLabel(yPos, IDC_STATIC_BASE + 530);
                    yPos += lineHeight;
                }

                // Padding slider (0 to 200, step 1, default 40)
                {
                    createLabel(L"Pad (px):", yPos, IDC_STATIC_BASE + 471);
                    createTrackbar(IDC_RECTIFY_PAD_PX, yPos, 0, 200, uiControls.rectifyPadPx);
                    createValueLabel(yPos, IDC_STATIC_BASE + 531);
                    yPos += lineHeight + gap;
                }

            } // End of rectifyExpanded block
        }

    }
    else {
        // ===== Display page: source selection =====
        addHeader(L"Source");

        // Refresh button
        {
            HWND hBtn = CreateWindowW(L"BUTTON", L"Refresh Camera List", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                      xPos, yPos - g_sidebarScrollPos, std::min(usableWidth, 180), 24,
                                      g_sidebarPanel, (HMENU)(INT_PTR)IDC_DISPLAY_REFRESH_CAMERAS, NULL, NULL);
            applyFont(hBtn, false);
        }
        yPos += lineHeight;

        HWND hCombo = CreateWindowW(L"COMBOBOX", L"", WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
                                    xPos, yPos - g_sidebarScrollPos, usableWidth, 200, g_sidebarPanel,
                                    (HMENU)(INT_PTR)IDC_DISPLAY_SOURCE_COMBO, NULL, NULL);
        if (hCombo) {
            applyFont(hCombo, false);
            SetWindowTheme(hCombo, L"Explorer", NULL);
            // Use SendMessageW explicitly: some toolchains/projects may not have UNICODE enabled,
            // and SendMessageA would interpret UTF-16 strings as ANSI and truncate at the first NUL.
            SendMessageW(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Test Image");
            SendMessageW(hCombo, CB_SETITEMDATA, 0, (LPARAM)kSourceTestImage);

            SendMessageW(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Test Video");
            SendMessageW(hCombo, CB_SETITEMDATA, 1, (LPARAM)kSourceTestVideo);

            int selectedIndex = 0;
            if (uiControls.selectedSource == kSourceTestVideo) selectedIndex = 1;
            int idx = 2;
            for (int cam : g_availableCameras) {
                std::wstring label = L"Camera " + std::to_wstring(cam);
                SendMessageW(hCombo, CB_ADDSTRING, 0, (LPARAM)label.c_str());
                SendMessageW(hCombo, CB_SETITEMDATA, idx, (LPARAM)cam);
                if (uiControls.selectedSource >= 0 && uiControls.selectedSource == cam) {
                    selectedIndex = idx;
                }
                idx++;
            }
            SendMessageW(hCombo, CB_SETCURSEL, selectedIndex, 0);
        }
        yPos += lineHeight + 10;

        {
            HWND h = CreateWindowW(L"STATIC", L"Note: camera opens after selection.", WS_VISIBLE | WS_CHILD | SS_LEFT,
                                   xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel, NULL, NULL, NULL);
            applyFont(h, false);
        }
        yPos += lineHeight;
    }
    
    // Scroll range covers the entire panel content.
    const int contentStartY = 0;
    // NOTE:
    // All debug trackbars are created inline within the feature groups above.
    // Display page still has no trackbars.
    
    // Store total content height for scrolling
    g_sidebarContentHeight = yPos - contentStartY + 10;
    
    // Container-level scrollbar (WS_VSCROLL on g_sidebarPanel)
    // Windows draws the scrollbar for us; we just set its range/page/pos.
    const int viewHeight = std::max(1, panelHeight - contentStartY);
    SCROLLINFO si{};
    si.cbSize = sizeof(SCROLLINFO);
    si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
    si.nMin = 0;
    si.nMax = std::max(0, g_sidebarContentHeight - 1);
    si.nPage = std::max(1, viewHeight);

    const int maxPos = std::max(si.nMin, (int)(si.nMax - (int)si.nPage + 1));
    g_sidebarScrollPos = std::max(si.nMin, std::min(g_sidebarScrollPos, maxPos));
    si.nPos = g_sidebarScrollPos;

    SetScrollInfo(g_sidebarPanel, SB_VERT, &si, TRUE);
    ShowScrollBar(g_sidebarPanel, SB_VERT, g_sidebarContentHeight > viewHeight);
    
    updateSidebarControls();
    
    // Force panel update to show controls
    if (g_sidebarPanel) {
        InvalidateRect(g_sidebarPanel, NULL, TRUE);
        UpdateWindow(g_sidebarPanel);
    }
}

// Destroy all sidebar controls
void destroySidebarControls() {
    // Win32-hosted UI: the sidebar panel persists; we only clear its children.
    if (g_sidebarPanel && IsWindow(g_sidebarPanel)) {
        EnumChildWindows(g_sidebarPanel, [](HWND hwndChild, LPARAM) -> BOOL {
            DestroyWindow(hwndChild);
            return TRUE;
        }, 0);
    }
    g_sidebarScrollPos = 0;
    g_sidebarContentHeight = 0;
}

// Update sidebar control values
void updateSidebarControls() {
    // Controls live under the sidebar panel.
    if (!g_sidebarPanel || !IsWindow(g_sidebarPanel)) return;
    
    // Update trackbar positions
    // Only update controls that exist (i.e., when their sections are expanded)
    HWND hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_COLOR_SENSITIVITY);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.colorSensitivity);
    
    // Diamond controls only exist when diamonds section is expanded
    if (uiControls.diamondsExpanded) {
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_SENSITIVITY);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.colorSensitivity);
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMONDS_ALPHA);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.fillAlpha);
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMONDS_THICKNESS);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.outlineThicknessPx);
    }
    
    // Rail controls only exist when rails section is expanded
    if (uiControls.railsExpanded) {
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_COLOR_SENSITIVITY);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.colorSensitivity);
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAILS_ALPHA);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.fillAlpha);
        hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAILS_THICKNESS);
        if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.outlineThicknessPx);
    }
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_ALPHA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.fillAlpha);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_THICKNESS);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.outlineThicknessPx);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RECTIFY_MARGIN_SCALE);
    if (hTrackbar) {
        int marginScaleInt = static_cast<int>(std::round(uiControls.rectifyMarginScale * 100.0f));
        marginScaleInt = std::clamp(marginScaleInt, 100, 140);
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, marginScaleInt);
    }
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RECTIFY_PAD_PX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.rectifyPadPx);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RECTIFY_MARGIN_SCALE);
    if (hTrackbar) {
        int marginScaleInt = static_cast<int>(std::round(uiControls.rectifyMarginScale * 100.0f));
        marginScaleInt = std::clamp(marginScaleInt, 100, 140);
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, marginScaleInt);
    }
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RECTIFY_PAD_PX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.rectifyPadPx);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_SMOOTHING);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.smoothingPercent);
    
    // Update value labels
    wchar_t buffer[32];

    // Felt color picker sensitivity value label (0..100)
    HWND hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 602);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.feltParams.colorSensitivity);
        SetWindowTextW(hLabel, buffer);
    }

    // Diamond color picker sensitivity value label (0..100) - only exists when diamonds section is expanded
    if (uiControls.diamondsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 502);
        if (hLabel) {
            swprintf_s(buffer, L"%d", uiControls.diamondParams.colorSensitivity);
            SetWindowTextW(hLabel, buffer);
        }
    }

    // Rail color picker sensitivity value label (0..100) - only exists when rails section is expanded
    if (uiControls.railsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 702);
        if (hLabel) {
            swprintf_s(buffer, L"%d", uiControls.railParams.colorSensitivity);
            SetWindowTextW(hLabel, buffer);
        }
    }

    // Felt color filtering values are presented via the Felt color picker labels (BGR/HSV/Range),
    // not via individual HSV sliders.

    // Alpha/thickness values
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 401);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.smoothingPercent); SetWindowTextW(hLabel, buffer); }

    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 117);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 119);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }
    // Rail outline thickness label - only exists when rails section is expanded
    if (uiControls.railsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 123);
        if (hLabel) { swprintf_s(buffer, L"%d", uiControls.railParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }
    }
    
    // Diamond outline thickness labels - only exist when diamonds section is expanded
    if (uiControls.diamondsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 124);
        if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 125);
        if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }
    }

    // Inline alpha value labels (next to Color buttons)
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 510);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    
    // Rail alpha label - only exists when rails section is expanded
    if (uiControls.railsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 522);
        if (hLabel) { swprintf_s(buffer, L"%d", uiControls.railParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    }
    
    // Diamond alpha label - only exists when diamonds section is expanded
    if (uiControls.diamondsExpanded) {
        hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 523);
        if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    }

    // Diamond detection count label (only exists when diamonds section is expanded)
    hLabel = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COUNT);
    if (hLabel && uiControls.diamondsExpanded) {
        swprintf_s(buffer, L"Detected: %d / 18", latestAnalysis.diamonds.numDiamondsDetected);
        SetWindowTextW(hLabel, buffer);
    }

    // The color picker info labels (BGR/HSV/Range) are STATIC controls; keep them in sync here so
    // rebuilding the sidebar immediately reflects the current picked colors/ranges for Felt.
    updateColorPickerLabels();
}

// Handle sidebar button clicks
void handleSidebarButton(int buttonId) {
    if (buttonId == IDC_SIDEBAR_COLLAPSE) {
        uiControls.sidebarCollapsed = !uiControls.sidebarCollapsed;
    }
    else if (buttonId == IDC_SIDEBAR_FELT) {
        uiControls.feltExpanded = !uiControls.feltExpanded;
        saveSettingsToDisk();
    }
    else if (buttonId == IDC_SIDEBAR_RAILS) {
        uiControls.railsExpanded = !uiControls.railsExpanded;
        saveSettingsToDisk();
    }
    else if (buttonId == IDC_SIDEBAR_DIAMONDS) {
        uiControls.diamondsExpanded = !uiControls.diamondsExpanded;
        saveSettingsToDisk();
    }
    else if (buttonId == IDC_SIDEBAR_RECTIFY) {
        uiControls.rectifyExpanded = !uiControls.rectifyExpanded;
        saveSettingsToDisk();
    }
}

// Update sidebar context display
void updateSidebarContext(SidebarContext context) {
    uiControls.sidebarContext = context;
    // UI refresh is driven by the Win32 host (layout + recreate controls).
}

// Handle trackbar value changes
void handleTrackbarChange(int trackbarId, int value) {
    switch (trackbarId) {
        case IDC_DIAMOND_COLOR_SENSITIVITY:
            uiControls.diamondParams.colorSensitivity = std::clamp(value, 0, 100);
            applyDiamondColorSensitivityToRangesFromPickedHSV();
            updateColorPickerLabels();
            break;
        case IDC_FELT_COLOR_SENSITIVITY:
            uiControls.feltParams.colorSensitivity = std::clamp(value, 0, 100);
            applyFeltColorSensitivityToRangesFromPickedHSV();
            updateColorPickerLabels();
            break;
        case IDC_RAIL_COLOR_SENSITIVITY:
            uiControls.railParams.colorSensitivity = std::clamp(value, 0, 100);
            applyRailColorSensitivityToRangesFromPickedHSV();
            updateColorPickerLabels();
            break;
        case IDC_DIAMONDS_ALPHA:
            uiControls.diamondParams.fillAlpha = value;
            break;
        case IDC_DIAMONDS_THICKNESS:
            uiControls.diamondParams.outlineThicknessPx = value;
            break;
        case IDC_FELT_ALPHA:
            uiControls.feltParams.fillAlpha = value;
            break;
        case IDC_FELT_THICKNESS:
            uiControls.feltParams.outlineThicknessPx = value;
            break;
        case IDC_RAILS_ALPHA:
            uiControls.railParams.fillAlpha = value;
            saveSettingsToDisk();
            break;
        case IDC_RAILS_THICKNESS:
            uiControls.railParams.outlineThicknessPx = value;
            break;
        case IDC_RECTIFY_MARGIN_SCALE:
            uiControls.rectifyMarginScale = value / 100.0f;
            // Update value label
            {
                HWND hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 530);
                if (hLabel) {
                    wchar_t buffer[32];
                    swprintf_s(buffer, L"%.2f", uiControls.rectifyMarginScale);
                    SetWindowTextW(hLabel, buffer);
                }
            }
            saveSettingsToDisk();
            break;
        case IDC_RECTIFY_PAD_PX:
            uiControls.rectifyPadPx = value;
            // Update value label
            {
                HWND hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 531);
                if (hLabel) {
                    wchar_t buffer[32];
                    swprintf_s(buffer, L"%d", uiControls.rectifyPadPx);
                    SetWindowTextW(hLabel, buffer);
                }
            }
            saveSettingsToDisk();
            break;
        case IDC_SMOOTHING:
            uiControls.smoothingPercent = value;
            break;
        default:
            break;
    }
    updateSidebarControls();
    // Save settings immediately when any parameter changes
    saveSettingsToDisk();
}

// Old drawSidebar function removed - replaced with native Windows controls

// Update overlay menu checkmarks
void updateOverlayMenu() {
    HMENU hMenu = GetMenu(g_hwnd);
    if (hMenu) {
        // Menu is top-level commands now. Check/uncheck them directly by ID.
        CheckMenuItem(hMenu, IDM_NAV_DEBUG,
                      (uiControls.sidebarPage == SidebarPage::Debug) ? MF_CHECKED : MF_UNCHECKED);
        CheckMenuItem(hMenu, IDM_NAV_DISPLAY,
                      (uiControls.sidebarPage == SidebarPage::Display) ? MF_CHECKED : MF_UNCHECKED);
    }
}

// Handle menu command
void handleMenuCommand(int menuId) {
    if (menuId == IDM_NAV_DEBUG) {
        uiControls.showSidebar = true;
        uiControls.sidebarPage = SidebarPage::Debug;
        updateOverlayMenu();
        layoutChildren(g_hwnd);
        destroySidebarControls();
        createSidebarControls(g_hwnd);
    }
    else if (menuId == IDM_NAV_DISPLAY) {
        uiControls.showSidebar = true;
        uiControls.sidebarPage = SidebarPage::Display;
        updateOverlayMenu();
        layoutChildren(g_hwnd);
        destroySidebarControls();
        createSidebarControls(g_hwnd);
    }
    else if (menuId == IDM_EXPORT_CAPTURES) {
        std::filesystem::path outDir;
        std::string err;
        const bool ok = exportCapturesToDisk(g_lastProcessedFrame, &outDir, &err);
#ifdef _WIN32
        std::wstring msg;
        if (ok) {
            msg = L"Captures exported to:\n" + outDir.wstring();
            MessageBoxW(g_hwnd, msg.c_str(), L"Export Captures", MB_OK | MB_ICONINFORMATION);
        } else {
            msg = L"Export failed.\n\nTarget:\n" + outDir.wstring() +
                  L"\n\nReason:\n" + std::wstring(err.begin(), err.end());
            MessageBoxW(g_hwnd, msg.c_str(), L"Export Captures", MB_OK | MB_ICONERROR);
        }
#endif
    }
}

// Sidebar container window procedure for handling scrolling
LRESULT CALLBACK SidebarContainerProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_VSCROLL: {
            int nScrollCode = LOWORD(wParam);
            int nPos = HIWORD(wParam);
            
            SCROLLINFO si = {0};
            si.cbSize = sizeof(SCROLLINFO);
            si.fMask = SIF_POS | SIF_RANGE | SIF_PAGE;
            GetScrollInfo(hwnd, SB_VERT, &si);
            
            int oldPos = si.nPos;
            int newPos = oldPos;
            
            switch (nScrollCode) {
                case SB_LINEUP:
                    newPos = oldPos - 20;
                    break;
                case SB_LINEDOWN:
                    newPos = oldPos + 20;
                    break;
                case SB_PAGEUP:
                    newPos = oldPos - (int)si.nPage;
                    break;
                case SB_PAGEDOWN:
                    newPos = oldPos + (int)si.nPage;
                    break;
                case SB_THUMBTRACK:
                case SB_THUMBPOSITION:
                    newPos = nPos;
                    break;
            }
            
            newPos = std::max(si.nMin, std::min(newPos, (int)(si.nMax - si.nPage + 1)));
            
            if (newPos != oldPos) {
                int delta = oldPos - newPos;
                ScrollWindowEx(hwnd, 0, delta, NULL, NULL, NULL, NULL, SW_INVALIDATE | SW_ERASE);
                si.nPos = newPos;
                si.fMask = SIF_POS;
                SetScrollInfo(hwnd, SB_VERT, &si, TRUE);
                g_sidebarScrollPos = newPos;
            }
            return 0;
        }
        
        case WM_HSCROLL: {
            // Forward trackbar messages to parent window
            return SendMessage(g_hwnd, WM_HSCROLL, wParam, lParam);
        }
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Window procedure to handle menu commands and trackbar messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_COMMAND) {
        int menuId = LOWORD(wParam);
        handleMenuCommand(menuId);
        return 0;
    }
    
    // Handle sidebar scrollbar
    if (uMsg == WM_VSCROLL && uiControls.showSidebar) {
        HWND hScrollbar = (HWND)lParam;
        if (hScrollbar == GetDlgItem(hwnd, IDC_STATIC_BASE + 100)) {
            int nScrollCode = LOWORD(wParam);
            int nPos = HIWORD(wParam);
            
            SCROLLINFO si = {0};
            si.cbSize = sizeof(SCROLLINFO);
            si.fMask = SIF_POS | SIF_RANGE | SIF_PAGE;
            GetScrollInfo(hScrollbar, SB_CTL, &si);
            
            int oldPos = si.nPos;
            int newPos = oldPos;
            
            switch (nScrollCode) {
                case SB_LINEUP:
                    newPos = oldPos - 20;
                    break;
                case SB_LINEDOWN:
                    newPos = oldPos + 20;
                    break;
                case SB_PAGEUP:
                    newPos = oldPos - (int)si.nPage;
                    break;
                case SB_PAGEDOWN:
                    newPos = oldPos + (int)si.nPage;
                    break;
                case SB_THUMBTRACK:
                case SB_THUMBPOSITION:
                    newPos = nPos;
                    break;
            }
            
            newPos = std::max(si.nMin, std::min(newPos, (int)(si.nMax - si.nPage + 1)));
            
            if (newPos != oldPos) {
                int delta = oldPos - newPos;
                g_sidebarScrollPos = newPos;
                
                // Move all sidebar controls
                RECT clientRect;
                GetClientRect(hwnd, &clientRect);
                int sidebarX = clientRect.right - SIDEBAR_WIDTH;
                
                EnumChildWindows(hwnd, [](HWND hwndChild, LPARAM lParam) -> BOOL {
                    RECT rect;
                    GetWindowRect(hwndChild, &rect);
                    POINT pt = {rect.left, rect.top};
                    ScreenToClient((HWND)lParam, &pt);
                    
                    RECT clientRect;
                    GetClientRect((HWND)lParam, &clientRect);
                    int sidebarX = clientRect.right - SIDEBAR_WIDTH;
                    
                    // If control is in sidebar area, move it
                    if (pt.x >= sidebarX) {
                        int* pDelta = (int*)lParam;
                        SetWindowPos(hwndChild, NULL, rect.left - GetSystemMetrics(SM_CXSCREEN) + 
                                   GetWindowRect((HWND)lParam, &rect) ? 0 : 0, 
                                   pt.y + *pDelta, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
                    }
                    return TRUE;
                }, (LPARAM)&delta);
                
                si.nPos = newPos;
                si.fMask = SIF_POS;
                SetScrollInfo(hScrollbar, SB_CTL, &si, TRUE);
                
                // Recreate controls with new scroll position (simpler approach)
                destroySidebarControls();
                createSidebarControls(hwnd);
            }
            return 0;
        }
    }
    
    // Handle trackbar (slider) changes
    if (uMsg == WM_HSCROLL) {
        HWND hTrackbar = (HWND)lParam;
        int trackbarId = GetDlgCtrlID(hTrackbar);
        int value = (int)SendMessage(hTrackbar, TBM_GETPOS, 0, 0);
        handleTrackbarChange(trackbarId, value);
        return 0;
    }
    
    // Handle window resize - update layout (containers)
    if (uMsg == WM_SIZE) {
        updateLayout(hwnd);
        if (uiControls.showSidebar) {
            // Recreate sidebar controls with new layout
            destroySidebarControls();
            createSidebarControls(hwnd);
        }
    }
    
    // Handle window close
    if (uMsg == WM_CLOSE) {
        // Let the default handler close the window
        // The main loop will detect this and exit
    }
    
    // Call original window procedure for other messages
    if (g_oldWndProc) {
        return CallWindowProc(g_oldWndProc, hwnd, uMsg, wParam, lParam);
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Enumerate available cameras
std::vector<int> enumerateCameras() {
    std::vector<int> cameras;
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture testCap(i);
        if (testCap.isOpened()) {
            cameras.push_back(i);
            testCap.release();
        }
    }
    return cameras;
}

// Actual program entry point:
// - On Windows: Win32 owns the main window; OpenCV only provides pixels.
// - Elsewhere: fall back to legacy HighGUI path.
int main(int argc, char** argv) {
#ifdef _WIN32
    return runWin32HostedApp(argc, argv);
#else
    return legacyHighGuiMain(argc, argv);
#endif
}

