#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <commctrl.h>  // For trackbar controls
#include <commdlg.h>   // For ChooseColor
#include <uxtheme.h>   // For SetWindowTheme (modern-ish control styling)
#include <algorithm>  // For std::max, std::min, std::sort
#pragma comment(lib, "comctl32.lib")  // Link against common controls library
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "uxtheme.lib")
#endif

#include "diamond_detection.h"
#include "felt_detection.h"
#include "rail_detection.h"
#include "orientation_detector.h"

// Sidebar context types
enum class SidebarContext {
    None,
    Diamonds,
    Felt,
    Rail
};

// Which *top-level* sidebar page is currently shown.
// This is driven by the top menu (Debug / Display).
enum class SidebarPage {
    Debug,
    Display
};

// Global variables for UI state
struct UIControls {
    bool showOverlay = true;  // Master overlay toggle (on by default)
    // Default all overlays ON so masks are visible immediately.
    bool showDiamonds = true;
    bool showFelt = true;
    bool showRail = true;
    // Orientation Mask: draws outer + inner quads bounding the rail mask, with labeled rail regions (L1, L2, S1, S2).
    bool showOrientation = false;
    bool showSidebar = true; // Debug sidebar toggle (on by default)
    bool sidebarCollapsed = false; // Sidebar collapsed state
    SidebarPage sidebarPage = SidebarPage::Debug; // Top-level sidebar page (default: Debug)
    SidebarContext sidebarContext = SidebarContext::Diamonds; // Current sidebar context (default to Diamonds)
    bool useTestImage = true;
    int selectedCamera = -1;
    int currentOpenedCamera = -1; // Track which camera is currently opened

    // Global rendering/UX tuning.
    // 0 = no smoothing; 100 = very strong smoothing (nearly "sticky").
    // We apply this as smoothing of the *overlay delta* (not the base video)
    // so the raw video stays crisp while the overlay stabilizes.
    int smoothingPercent = 0;
    
    // Overlay parameters (defined in respective detection headers)
    DiamondDetectionParams diamondParams;
    FeltParams feltParams;
    RailParams railParams;
} uiControls;

// Last rendered (overlaid) frame so menu-driven capture export can work.
static cv::Mat g_lastProcessedFrame;
static cv::Mat g_lastSourceFrame;  // Original unscaled source frame for color picking

static TableOrientation g_tableOrientation;

// Last diamond detection processing image (the final image used for blob detection)
static cv::Mat g_lastDiamondProcessingImage;
// Additional debug images emitted by diamond detection (label, image).
extern std::vector<std::pair<std::string, cv::Mat>> g_lastDiamondDebugImages;

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

    // Always attempt overlay first.
    writeImageChecked(dir / (stem + "-overlay.png"), processedImage);

    // Export diamond detection processing image if available
    if (!g_lastDiamondProcessingImage.empty()) {
        writeImageChecked(dir / (stem + "-diamond-processing.png"), g_lastDiamondProcessingImage);
    }
    // Export per-stage diamond debug images if available
    if (!g_lastDiamondDebugImages.empty()) {
        for (const auto& kv : g_lastDiamondDebugImages) {
            std::string label = kv.first;
            // sanitize label for filename
            for (char& c : label) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) c = '_';
            }
            writeImageChecked(dir / (stem + "-diamond-" + label + ".png"), kv.second);
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

// Color picker info labels (defined early for use in update function)
#define IDC_COLOR_PICKER_BGR 10050
#define IDC_COLOR_PICKER_HSV 10051
#define IDC_COLOR_PICKER_RANGE_H 10052
#define IDC_COLOR_PICKER_RANGE_S 10053
#define IDC_COLOR_PICKER_RANGE_V 10054
#define IDC_COLOR_PICKER_SWATCH 10055

static HWND g_imageViewHwnd = NULL;
static HWND g_zoomWindowHwnd = NULL;  // Zoom window for color picker
static HBRUSH g_imageBgBrush = NULL;
static HBRUSH g_sidebarBgBrush = NULL;
// Color picker swatch brush:
// - The swatch is a STATIC control. To paint it correctly and reliably, we provide a brush via
//   WM_CTLCOLORSTATIC from the SidebarPanelProc.
// - We own this brush and must delete it on replacement / shutdown to avoid leaking GDI objects.
static HBRUSH g_colorPickerSwatchBrush = NULL;
static HFONT g_sidebarFont = NULL;
static HFONT g_sidebarFontBold = NULL;
static bool g_colorPickerActive = false;  // Whether color picker mode is active
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

// App data sources
static cv::Mat g_testImage;
static cv::VideoCapture g_camera;

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
// - `g_requestedCameraIndex`: -1 means "Test Image", >= 0 means "open that camera index"
// - The capture thread updates `g_latestCameraFrame` whenever it successfully reads a frame.
// - The UI thread renders:
//   - test image directly when `useTestImage == true` (never waits on camera thread)
//   - else: latest camera frame if available, otherwise test image as fallback.
// ------------------------------------------------------------------------------------------------
static std::atomic<int> g_requestedCameraIndex{-1};
static std::atomic<bool> g_captureThreadRunning{false};
static std::thread g_captureThread;
static std::mutex g_latestFrameMutex;
static cv::Mat g_latestCameraFrame;

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
        int currentCam = -2; // sentinel; forces first open attempt if requested >= 0

        while (g_captureThreadRunning.load(std::memory_order_relaxed)) {
            const int requested = g_requestedCameraIndex.load(std::memory_order_relaxed);

            // Switch camera / release camera as needed.
            if (requested != currentCam) {
                if (cap.isOpened()) {
                    cap.release();
                }

                currentCam = requested;

                if (currentCam >= 0) {
                    // Prefer a backend that tends to behave well on Windows.
                    // If this fails, OpenCV will fall back internally on some installs;
                    // but using an explicit backend often reduces open latency/hangs.
                    cap.open(currentCam, cv::CAP_DSHOW);
                    if (cap.isOpened()) {
                        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                    }
                }
            }

            // If no camera requested, just idle.
            if (currentCam < 0 || !cap.isOpened()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            cv::Mat frame;
            // NOTE: Some backends block in read(). If that happens, UI still stays responsive because
            // this thread is the only one affected. UI can always switch back to Test Image.
            if (!cap.read(frame) || frame.empty()) {
                // Backoff a bit on read failure.
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(g_latestFrameMutex);
                g_latestCameraFrame = frame;
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
static void applyColorSensitivityToRangesFromPickedHSV();
static void updateImageDibFromBgr(const cv::Mat& bgr);

// Update the global swatch brush used by WM_CTLCOLORSTATIC.
static void setColorPickerSwatchBrush(COLORREF swatchColor) {
    if (g_colorPickerSwatchBrush) {
        DeleteObject(g_colorPickerSwatchBrush);
        g_colorPickerSwatchBrush = NULL;
    }
    g_colorPickerSwatchBrush = CreateSolidBrush(swatchColor);
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
            if (child && GetDlgCtrlID(child) == IDC_COLOR_PICKER_SWATCH && g_colorPickerSwatchBrush) {
                // It's a solid color block: no text, so just set a matching background.
                // (Text color doesn't matter because the swatch has an empty caption.)
                SetBkMode(hdc, OPAQUE);
                return (INT_PTR)g_colorPickerSwatchBrush;
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
    if (!g_colorPickerActive || g_lastSourceFrame.empty()) {
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
    if (!g_colorPickerActive || g_lastSourceFrame.empty()) return;
    
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
        
        // Store the picked BGR color
        uiControls.diamondParams.pickedBGR = bgr;
        uiControls.diamondParams.pickedHSV = hsv;
        uiControls.diamondParams.hasPickedColor = true;
        
        // Set HSV ranges around the sampled color based on the Sensitivity slider.
        applyColorSensitivityToRangesFromPickedHSV();
        
        uiControls.diamondParams.use_color_filter = true;

        // Hide magnifier and deactivate picker FIRST, then refresh UI.
        // `updateColorPickerLabels()` uses `g_colorPickerActive` to decide the button title.
        if (g_magnifierHwnd) {
            ShowWindow(g_magnifierHwnd, SW_HIDE);
        }
        g_colorPickerActive = false;

        // Update UI labels (also updates button text)
        updateColorPickerLabels();
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

            if (g_imageDib.width > 0 && g_imageDib.height > 0 && !g_imageDib.bgra.empty()) {
                SetStretchBltMode(memDC, HALFTONE);

                StretchDIBits(
                    memDC,
                    0, 0, dstW, dstH,
                    0, 0, g_imageDib.width, g_imageDib.height,
                    g_imageDib.bgra.data(),
                    &g_imageDib.bmi,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
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
            if (g_colorPickerActive && !g_lastSourceFrame.empty()) {
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
                // Then clamp to g_lastSourceFrame dimensions (source image we're sampling from)
                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                    // Scale coordinates back to displayed image, then to source image
                    imgX = (int)((double)x / dstW * g_imageDib.width);
                    imgY = (int)((double)y / dstH * g_imageDib.height);
                    // Clamp to source frame dimensions
                    imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                    imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
                } else {
                    imgX = std::clamp(x, 0, g_lastSourceFrame.cols - 1);
                    imgY = std::clamp(y, 0, g_lastSourceFrame.rows - 1);
                }
                
                // Get screen coordinates for positioning zoom window
                POINT pt = {x, y};
                ClientToScreen(hwnd, &pt);
                
                updateMagnifierWindow(imgX, imgY, pt.x, pt.y);
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
            if (g_colorPickerActive && !g_lastSourceFrame.empty()) {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);
                
                // Convert client coordinates to source image coordinates
                RECT rc{};
                GetClientRect(hwnd, &rc);
                const int dstW = std::max(0L, rc.right - rc.left);
                const int dstH = std::max(0L, rc.bottom - rc.top);
                
                int imgX, imgY;
                // Use g_imageDib dimensions for coordinate conversion (displayed image)
                // Then clamp to g_lastSourceFrame dimensions (source image we're sampling from)
                if (dstW > 0 && dstH > 0 && g_imageDib.width > 0 && g_imageDib.height > 0) {
                    // Scale coordinates back to displayed image, then to source image
                    imgX = (int)((double)x / dstW * g_imageDib.width);
                    imgY = (int)((double)y / dstH * g_imageDib.height);
                    // Clamp to source frame dimensions
                    imgX = std::clamp(imgX, 0, g_lastSourceFrame.cols - 1);
                    imgY = std::clamp(imgY, 0, g_lastSourceFrame.rows - 1);
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
                    
                    // Store the picked BGR color
                    uiControls.diamondParams.pickedBGR = bgr;
                    uiControls.diamondParams.pickedHSV = hsv;
                    uiControls.diamondParams.hasPickedColor = true;
                    
                    // Set HSV ranges around the sampled color based on the Sensitivity slider.
                    applyColorSensitivityToRangesFromPickedHSV();
                    
                    uiControls.diamondParams.use_color_filter = true;
                    
                    // Hide magnifier and deactivate picker FIRST, then refresh UI.
                    // `updateColorPickerLabels()` uses `g_colorPickerActive` to decide the button title.
                    if (g_magnifierHwnd) {
                        ShowWindow(g_magnifierHwnd, SW_HIDE);
                    }
                    g_colorPickerActive = false;

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
    
    wchar_t bgrText[64] = L"BGR: --";
    wchar_t hsvText[64] = L"HSV: --";
    wchar_t rangeText[128] = L"Range: --";
    
    // Color filtering is now always enabled; we always show the current picked color and ranges.
    {
        cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
        swprintf_s(bgrText, L"BGR: (%d, %d, %d)",
                  (int)bgr[2], (int)bgr[1], (int)bgr[0]);

        if (uiControls.diamondParams.hasPickedColor) {
            const cv::Vec3b hsv = uiControls.diamondParams.pickedHSV;
            swprintf_s(hsvText, L"HSV: (%d, %d, %d)", (int)hsv[0], (int)hsv[1], (int)hsv[2]);
        }

        // If hue is wrapped, we intentionally store HMin > HMax to represent:
        //   H in [0..HMax] U [HMin..180]
        if (uiControls.diamondParams.colorHMin <= uiControls.diamondParams.colorHMax) {
            swprintf_s(rangeText, L"Range: H[%d-%d] S[%d-%d] V[%d-%d]",
                      uiControls.diamondParams.colorHMin, uiControls.diamondParams.colorHMax,
                      uiControls.diamondParams.colorSMin, uiControls.diamondParams.colorSMax,
                      uiControls.diamondParams.colorVMin, uiControls.diamondParams.colorVMax);
        } else {
            swprintf_s(rangeText, L"Range: H[0-%d] U [%d-180] S[%d-%d] V[%d-%d]",
                      uiControls.diamondParams.colorHMax,
                      uiControls.diamondParams.colorHMin,
                      uiControls.diamondParams.colorSMin, uiControls.diamondParams.colorSMax,
                      uiControls.diamondParams.colorVMin, uiControls.diamondParams.colorVMax);
        }
    }
    
    HWND hBGR = GetDlgItem(g_sidebarPanel, IDC_COLOR_PICKER_BGR);
    if (hBGR) SetWindowTextW(hBGR, bgrText);
    
    HWND hHSV = GetDlgItem(g_sidebarPanel, IDC_COLOR_PICKER_HSV);
    if (hHSV) SetWindowTextW(hHSV, hsvText);
    
    HWND hRange = GetDlgItem(g_sidebarPanel, IDC_COLOR_PICKER_RANGE_H);
    if (hRange) SetWindowTextW(hRange, rangeText);
    
    // Update color swatch
    HWND hSwatch = GetDlgItem(g_sidebarPanel, IDC_COLOR_PICKER_SWATCH);
    if (hSwatch) {
        cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
        COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]);
        setColorPickerSwatchBrush(swatchColor);
        InvalidateRect(hSwatch, NULL, TRUE);
        UpdateWindow(hSwatch);
    }
    
    // Update button text
    HWND hColorPicker = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_PICKER);
    if (hColorPicker) {
        const wchar_t* btnText = g_colorPickerActive ? L"Cancel (Click to Pick)" : L"Pick Diamond Color";
        SetWindowTextW(hColorPicker, btnText);
    }
}

// Compute HSV tolerances based on the user-facing sensitivity (0..100).
// - 0   => tight match (strict)
// - 100 => loose match (more variation)
static void applyColorSensitivityToRangesFromPickedHSV() {
    if (!uiControls.diamondParams.hasPickedColor) return;

    const int sens = std::clamp(uiControls.diamondParams.colorSensitivity, 0, 100);
    const float t = static_cast<float>(sens) / 100.0f;

    // Tuned to be usable for typical consumer cameras / mixed lighting.
    // Hue is 0..180 in OpenCV. Saturation/Value are 0..255.
    const int hTol = (int)std::lround(5.0f + t * 25.0f);    // 5..30
    const int sTol = (int)std::lround(10.0f + t * 80.0f);   // 10..90
    const int vTol = (int)std::lround(10.0f + t * 100.0f);  // 10..110

    const cv::Vec3b hsv = uiControls.diamondParams.pickedHSV;
    const int h = (int)hsv[0];
    const int s = (int)hsv[1];
    const int v = (int)hsv[2];

    // Hue wrap: represent wrap by setting HMin > HMax (handled in diamond_detection.cpp).
    const int hMinRaw = h - hTol;
    const int hMaxRaw = h + hTol;
    if (hMinRaw < 0 || hMaxRaw > 180) {
        const int hMinWrap = (hMinRaw < 0) ? (180 + hMinRaw) : hMinRaw;
        const int hMaxWrap = (hMaxRaw > 180) ? (hMaxRaw - 180) : hMaxRaw;
        uiControls.diamondParams.colorHMin = std::clamp(hMinWrap, 0, 180);
        uiControls.diamondParams.colorHMax = std::clamp(hMaxWrap, 0, 180);
        // Note: HMin > HMax means wrapped.
    } else {
        uiControls.diamondParams.colorHMin = std::clamp(hMinRaw, 0, 180);
        uiControls.diamondParams.colorHMax = std::clamp(hMaxRaw, 0, 180);
    }

    uiControls.diamondParams.colorSMin = std::clamp(s - sTol, 0, 255);
    uiControls.diamondParams.colorSMax = std::clamp(s + sTol, 0, 255);
    uiControls.diamondParams.colorVMin = std::clamp(v - vTol, 0, 255);
    uiControls.diamondParams.colorVMax = std::clamp(v + vTol, 0, 255);
}

static void layoutChildren(HWND mainHwnd) {
    RECT rc{};
    GetClientRect(mainHwnd, &rc);
    const int w = (int)(rc.right - rc.left);
    const int h = (int)(rc.bottom - rc.top);

    const int sidebarW =
        (uiControls.showSidebar && !uiControls.sidebarCollapsed) ? 300 :
        (uiControls.showSidebar && uiControls.sidebarCollapsed) ? 30 : 0;

    const int imageW = std::max(0, w - sidebarW);
    const int imageH = std::max(0, h);

    if (g_imageViewHwnd) {
        MoveWindow(g_imageViewHwnd, 0, 0, imageW, imageH, TRUE);
    }
    if (g_sidebarPanel) {
        if (uiControls.showSidebar) {
            ShowWindow(g_sidebarPanel, SW_SHOW);
            MoveWindow(g_sidebarPanel, imageW, 0, sidebarW, imageH, TRUE);
        } else {
            ShowWindow(g_sidebarPanel, SW_HIDE);
        }
    }
}

// NOTE: Orientation logic (table-axis estimation + Orientation Mask overlay) lives in
// `src/orientation_detector.{h,cpp}`. main.cpp should only own UI wiring and high-level calls.

// Minimal per-frame render: apply overlays (using OpenCV), then push to ImageView.
// We keep a small amount of global state to support temporal smoothing.
static cv::Mat g_prevOverlayDeltaF; // CV_32FC3: (processed - rawFrame) from previous frame
static cv::Mat buildDisplayFrame(const cv::Mat& currentFrame) {
    if (currentFrame.empty()) return {};
    cv::Mat processed = currentFrame.clone();

    if (uiControls.showOverlay) {
        // Felt contour is needed for both felt and rail overlays (and rail detection).
        std::vector<cv::Point> feltContour;
        if (uiControls.showFelt || uiControls.showRail || uiControls.showOrientation) {
            feltContour = detectFeltContour(currentFrame, uiControls.feltParams);
        }

        // Rail mask is needed for both rail overlay AND the orientation overlay.
        cv::Mat railMask;
        if ((uiControls.showRail || uiControls.showOrientation) && !feltContour.empty()) {
            railMask = detectRailMask(currentFrame, feltContour, uiControls.railParams);

            // Compute table orientation (two dominant axes) from rail mask.
            g_tableOrientation = computeTableOrientationFromRailMask(railMask);
        }

        if (uiControls.showFelt && !feltContour.empty()) {
            std::vector<std::vector<cv::Point>> contours{feltContour};
            if (uiControls.feltParams.isFilled) {
                cv::Mat overlay = processed.clone();
                cv::fillPoly(overlay, contours, uiControls.feltParams.color);
                const double a = std::clamp(uiControls.feltParams.fillAlpha, 0, 255) / 255.0;
                cv::addWeighted(processed, 1.0 - a, overlay, a, 0, processed);
                // Outline for definition
                cv::drawContours(processed, contours, -1, uiControls.feltParams.color, std::max(1, uiControls.feltParams.outlineThicknessPx));
            } else {
                cv::drawContours(processed, contours, -1, uiControls.feltParams.color, std::max(1, uiControls.feltParams.outlineThicknessPx));
            }
        }

        if (uiControls.showRail && !railMask.empty() && cv::countNonZero(railMask) > 0) {
                if (uiControls.railParams.isFilled) {
                    cv::Mat overlay = processed.clone();
                    overlay.setTo(uiControls.railParams.color, railMask);
                    const double a = std::clamp(uiControls.railParams.fillAlpha, 0, 255) / 255.0;
                    cv::addWeighted(processed, 1.0 - a, overlay, a, 0, processed);
                } else {
                    // Outline-only mode: draw the actual mask contour (not a fitted box).
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(railMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    if (!contours.empty()) {
                        cv::drawContours(processed, contours, -1, uiControls.railParams.color, std::max(1, uiControls.railParams.outlineThicknessPx));
                    }
                }
        }

        // Orientation Mask overlay (geometry + labeling) lives in `orientation_detector`.
        if (uiControls.showOrientation && !railMask.empty() && cv::countNonZero(railMask) > 0) {
            OrientationMaskRenderParams p;
            p.lineColorBGR = cv::Scalar(255, 0, 255);
            p.fillColorBGR = cv::Scalar(255, 0, 255);
            p.lineThicknessPx = 5;
            p.fillAlpha = 0.30;
            p.innerShrinkFraction = 0.05f;
            p.labelFontScale = 1.5;
            p.labelThicknessPx = 3;
            drawOrientationMaskOverlay(processed, railMask, feltContour, p);
        }

        if (uiControls.showDiamonds) {
            detectDiamonds(currentFrame, processed, true, uiControls.diamondParams, uiControls.feltParams, uiControls.railParams, &g_lastDiamondProcessingImage);
        } else {
            g_lastDiamondProcessingImage.release();
        }
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

    // Always render test image immediately when selected (never wait for camera I/O).
    if (uiControls.useTestImage) {
        frame = g_testImage;
        g_requestedCameraIndex.store(-1, std::memory_order_relaxed);
    } else {
        g_requestedCameraIndex.store(uiControls.selectedCamera, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(g_latestFrameMutex);
            frame = g_latestCameraFrame;
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
            if (id == kIdSidebarCollapse || id == kIdSidebarDiamonds || id == kIdSidebarFelt || id == kIdSidebarRail) {
                // Reuse existing handler
                handleSidebarButton(id);
                layoutChildren(hwnd);
                createSidebarControls(hwnd);
                return 0;
            }

            // Debug sidebar checkboxes (IDs match defines below: 30200..30203)
            const int kIdOverlayMaster = 30200;
            const int kIdOverlayDiamonds = 30201;
            const int kIdOverlayFelt = 30202;
            const int kIdOverlayRail = 30203;
            const int kIdOverlayOrientation = 30204;
            if (code == BN_CLICKED && (id == kIdOverlayMaster || id == kIdOverlayDiamonds || id == kIdOverlayFelt || id == kIdOverlayRail || id == kIdOverlayOrientation)) {
                const bool isChecked = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                if (id == kIdOverlayMaster) uiControls.showOverlay = isChecked;
                else if (id == kIdOverlayDiamonds) uiControls.showDiamonds = isChecked;
                else if (id == kIdOverlayFelt) uiControls.showFelt = isChecked;
                else if (id == kIdOverlayRail) uiControls.showRail = isChecked;
                else if (id == kIdOverlayOrientation) uiControls.showOrientation = isChecked;
                return 0;
            }

            // Overlay style buttons/checkboxes (IDs match defines below: 30220+)
            const int kIdDiamondsColor = 30220;
            const int kIdDiamondsFilled = 30221;
            const int kIdFeltColor = 30230;
            const int kIdFeltFilled = 30231;
            const int kIdRailColor = 30240;
            const int kIdRailFilled = 30241;
            const int kIdDiamondSkipMorph = 30250;  // IDC_DIAMOND_SKIP_MORPH
            if (code == BN_CLICKED) {
                if (id == kIdDiamondsFilled) {
                    uiControls.diamondParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    return 0;
                }
                if (id == kIdFeltFilled) {
                    uiControls.feltParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    return 0;
                }
                if (id == kIdRailFilled) {
                    uiControls.railParams.isFilled = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    return 0;
                }
                if (id == kIdDiamondSkipMorph) {
                    uiControls.diamondParams.skip_morph_enhancement = (SendMessage(hwndCtl, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    return 0;
                }
                if (id == IDC_DIAMOND_COLOR_PICKER) {
                    // Activate/deactivate color picker mode
                    g_colorPickerActive = !g_colorPickerActive;
                    if (!g_colorPickerActive) {
                        // Hide magnifier when deactivating
                        if (g_magnifierHwnd) {
                            ShowWindow(g_magnifierHwnd, SW_HIDE);
                        }
                    } else {
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
                    }
                    return 0;
                }
                if (id == kIdFeltColor) {
                    cv::Scalar newColor = uiControls.feltParams.color;
                    if (chooseColor(hwnd, uiControls.feltParams.color, newColor)) {
                        uiControls.feltParams.color = newColor;
                    }
                    return 0;
                }
                if (id == kIdRailColor) {
                    cv::Scalar newColor = uiControls.railParams.color;
                    if (chooseColor(hwnd, uiControls.railParams.color, newColor)) {
                        uiControls.railParams.color = newColor;
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
                        uiControls.useTestImage = true;
                        uiControls.selectedCamera = -1;
                        g_requestedCameraIndex.store(-1, std::memory_order_relaxed);
                    } else {
                        const LRESULT camData = SendMessage(hwndCtl, CB_GETITEMDATA, sel, 0);
                        if (camData == CB_ERR) {
                            // Fallback to Test Image if item-data is unavailable.
                            uiControls.useTestImage = true;
                            uiControls.selectedCamera = -1;
                            g_requestedCameraIndex.store(-1, std::memory_order_relaxed);
                            return 0;
                        }
                        const int cam = (int)camData;
                        uiControls.useTestImage = false;
                        uiControls.selectedCamera = cam;
                        g_requestedCameraIndex.store(cam, std::memory_order_relaxed);
                    }
                }
                return 0;
            }

            // Refresh camera list button (IDC_BUTTON_BASE + 280 = 30280)
            if (code == BN_CLICKED && id == 30280) {
                // Hint to capture thread to release camera before probing devices.
                g_requestedCameraIndex.store(-1, std::memory_order_relaxed);
                uiControls.useTestImage = true;
                uiControls.selectedCamera = -1;
                uiControls.currentOpenedCamera = -1;

                g_availableCameras = enumerateCameras();

                // If current selection no longer exists, fall back to Test Image.
                if (!uiControls.useTestImage) {
                    bool found = false;
                    for (int cam : g_availableCameras) {
                        if (cam == uiControls.selectedCamera) { found = true; break; }
                    }
                    if (!found) {
                        uiControls.useTestImage = true;
                        uiControls.selectedCamera = -1;
                    }
                }

                // Rebuild sidebar (Display page shows the refreshed list)
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
            if (uiControls.showSidebar) createSidebarControls(hwnd);
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
            if (g_colorPickerSwatchBrush) {
                DeleteObject(g_colorPickerSwatchBrush);
                g_colorPickerSwatchBrush = NULL;
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

    // Start capture thread with current selection (default is Test Image).
    g_requestedCameraIndex.store(-1, std::memory_order_relaxed);
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
#define IDC_SIDEBAR_RAIL (IDC_BUTTON_BASE + 103)

// Control IDs for sidebar controls
#define IDC_STATIC_BASE 10000
#define IDC_TRACKBAR_BASE 20000
#define IDC_BUTTON_BASE 30000
#define IDC_COMBO_BASE 40000

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
#define IDC_RAIL_BLACK_VMAX (IDC_TRACKBAR_BASE + 10)
#define IDC_RAIL_BROWN_HMAX (IDC_TRACKBAR_BASE + 11)
#define IDC_RAIL_BROWN_SMAX (IDC_TRACKBAR_BASE + 12)
#define IDC_RAIL_BROWN_VMAX (IDC_TRACKBAR_BASE + 13)

// Button IDs
#define IDC_DIAMOND_COLOR (IDC_BUTTON_BASE + 1)
#define IDC_FELT_COLOR (IDC_BUTTON_BASE + 2)
#define IDC_RAIL_COLOR (IDC_BUTTON_BASE + 3)

// Sidebar checkbox IDs (Debug page)
#define IDC_DEBUG_OVERLAY_MASTER (IDC_BUTTON_BASE + 200)
#define IDC_DEBUG_OVERLAY_DIAMONDS_CB (IDC_BUTTON_BASE + 201)
#define IDC_DEBUG_OVERLAY_FELT_CB (IDC_BUTTON_BASE + 202)
#define IDC_DEBUG_OVERLAY_RAIL_CB (IDC_BUTTON_BASE + 203)
#define IDC_DEBUG_OVERLAY_ORIENTATION_CB (IDC_BUTTON_BASE + 204)

// Overlay style controls
#define IDC_DIAMONDS_STYLE_COLOR (IDC_BUTTON_BASE + 220)
#define IDC_DIAMONDS_STYLE_FILLED (IDC_BUTTON_BASE + 221)
#define IDC_FELT_STYLE_COLOR (IDC_BUTTON_BASE + 230)
#define IDC_FELT_STYLE_FILLED (IDC_BUTTON_BASE + 231)
#define IDC_RAIL_STYLE_COLOR (IDC_BUTTON_BASE + 240)
#define IDC_RAIL_STYLE_FILLED (IDC_BUTTON_BASE + 241)

// Sidebar combobox IDs (Display page)
#define IDC_DISPLAY_SOURCE_COMBO (IDC_COMBO_BASE + 1)
#define IDC_DISPLAY_REFRESH_CAMERAS (IDC_BUTTON_BASE + 280)

// Felt parameter trackbars
#define IDC_FELT_BLUE_HMIN (IDC_TRACKBAR_BASE + 50)
#define IDC_FELT_BLUE_HMAX (IDC_TRACKBAR_BASE + 51)
#define IDC_FELT_BLUE_SMIN (IDC_TRACKBAR_BASE + 52)
#define IDC_FELT_BLUE_VMIN (IDC_TRACKBAR_BASE + 53)
#define IDC_FELT_GREEN_HMIN (IDC_TRACKBAR_BASE + 54)
#define IDC_FELT_GREEN_HMAX (IDC_TRACKBAR_BASE + 55)
#define IDC_FELT_GREEN_SMIN (IDC_TRACKBAR_BASE + 56)
#define IDC_FELT_GREEN_VMIN (IDC_TRACKBAR_BASE + 57)

// Overlay style trackbars
#define IDC_DIAMONDS_RADIUS (IDC_TRACKBAR_BASE + 60)
#define IDC_DIAMONDS_THICKNESS (IDC_TRACKBAR_BASE + 61)
#define IDC_DIAMONDS_ALPHA (IDC_TRACKBAR_BASE + 62)
#define IDC_FELT_ALPHA (IDC_TRACKBAR_BASE + 70)
#define IDC_FELT_THICKNESS (IDC_TRACKBAR_BASE + 71)
#define IDC_RAIL_ALPHA (IDC_TRACKBAR_BASE + 80)
#define IDC_RAIL_THICKNESS (IDC_TRACKBAR_BASE + 81)

// Global tuning trackbars
#define IDC_SMOOTHING (IDC_TRACKBAR_BASE + 95)

// Legacy HighGUI path (kept around for reference/fallback). We do NOT use this on Windows anymore.
int legacyHighGuiMain(int argc, char** argv) {
    // Enumerate available cameras
    g_availableCameras = enumerateCameras();
    
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
        
        // Get current frame (test image or camera)
        cv::Mat currentFrame;
        if (uiControls.useTestImage) {
            // Close camera if we're switching to test image
            if (camera.isOpened()) {
                camera.release();
                uiControls.currentOpenedCamera = -1;
            }
            testImage.copyTo(currentFrame);
        } else {
            // Check if we need to switch cameras
            if (uiControls.selectedCamera >= 0) {
                // If camera is open but different camera selected, close and reopen
                if (camera.isOpened() && uiControls.currentOpenedCamera != uiControls.selectedCamera) {
                    camera.release();
                    uiControls.currentOpenedCamera = -1;
                }
                
                // Open camera if not already open
                if (!camera.isOpened()) {
                    camera.open(uiControls.selectedCamera);
                    if (camera.isOpened()) {
                        uiControls.currentOpenedCamera = uiControls.selectedCamera;
                        // Set some camera properties for better performance
                        camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                    } else {
                        std::cerr << "Warning: Could not open camera " << uiControls.selectedCamera << std::endl;
                        uiControls.useTestImage = true; // Fall back to test image
                        uiControls.selectedCamera = -1;
                    }
                }
            }
            
            if (camera.isOpened()) {
                camera >> currentFrame;
                if (currentFrame.empty()) {
                    testImage.copyTo(currentFrame); // Fallback if camera fails
                }
            } else {
                testImage.copyTo(currentFrame);
            }
        }
        
        // Process the image
        cv::Mat processedImage;
        currentFrame.copyTo(processedImage);
        
        // Apply overlays only if master overlay toggle is enabled
        // Master toggle acts as a gate - individual toggles control what's shown
        if (uiControls.showOverlay) {
            // Detect felt contour first (needed for rail and diamond detection)
            std::vector<cv::Point> feltContour = detectFeltContour(currentFrame);
            cv::Rect feltRect = feltContour.empty() ? cv::Rect(0, 0, currentFrame.cols, currentFrame.rows) 
                                                     : cv::boundingRect(feltContour);
            
            // Apply felt detection overlay if enabled
            if (uiControls.showFelt) {
                if (!feltContour.empty()) {
                    // Draw the detailed perimeter contour
                    std::vector<std::vector<cv::Point>> contours;
                    contours.push_back(feltContour);
                    cv::drawContours(processedImage, contours, -1, cv::Scalar(0, 255, 0), 2);
                    
                    // Also draw a semi-transparent overlay
                    cv::Mat overlay = processedImage.clone();
                    cv::fillPoly(overlay, contours, cv::Scalar(0, 255, 0));
                    cv::addWeighted(processedImage, 0.7, overlay, 0.3, 0, processedImage);
                }
            }
            
            // Apply rail detection overlay if enabled
            if (uiControls.showRail) {
                if (!feltContour.empty()) {
                    std::vector<std::vector<cv::Point>> railContours = detectRailContours(currentFrame, feltContour);
                    if (!railContours.empty()) {
                        // Draw rail contours with orange/brown color
                        cv::drawContours(processedImage, railContours, -1, cv::Scalar(0, 165, 255), 2);
                        
                        // Draw semi-transparent overlay for rail areas
                        cv::Mat overlay = processedImage.clone();
                        cv::fillPoly(overlay, railContours, cv::Scalar(0, 165, 255));
                        cv::addWeighted(processedImage, 0.7, overlay, 0.3, 0, processedImage);
                    } else {
                        // Fallback: Draw the rail mask directly if contours are empty
                        cv::Mat railMask = detectRailMask(currentFrame, feltContour);
                        if (cv::countNonZero(railMask) > 0) {
                            cv::Mat overlay = processedImage.clone();
                            overlay.setTo(cv::Scalar(0, 165, 255), railMask); // Blue overlay for rail
                            cv::addWeighted(processedImage, 0.7, overlay, 0.3, 0, processedImage);
                        }
                    }
                }
            }
            
            // Apply diamond detection if enabled
            if (uiControls.showDiamonds) {
                detectDiamonds(currentFrame, processedImage, true, uiControls.diamondParams, uiControls.feltParams, uiControls.railParams, &g_lastDiamondProcessingImage);
            } else {
                g_lastDiamondProcessingImage.release();
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
        // ===== Debug page: Global + 3 feature groups =====
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

        // Orientation overlay toggle
        {
            HWND h = CreateWindowW(L"BUTTON", L"Orientation Mask", WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                   xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel,
                                   (HMENU)(INT_PTR)IDC_DEBUG_OVERLAY_ORIENTATION_CB, NULL, NULL);
            applyFont(h, false);
            SendMessage(h, BM_SETCHECK, uiControls.showOrientation ? BST_CHECKED : BST_UNCHECKED, 0);
        }
        yPos += lineHeight;

        // Global smoothing
        createLabel(L"Smoothing:", yPos, IDC_STATIC_BASE + 400);
        createTrackbar(IDC_SMOOTHING, yPos, 0, 100, uiControls.smoothingPercent);
        createValueLabel(yPos, IDC_STATIC_BASE + 401);
        yPos += lineHeight + gap;

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

        // FELT group
        {
            addFeatureGroup(L"Felt",
                            IDC_DEBUG_OVERLAY_FELT_CB, uiControls.showFelt,
                            IDC_FELT_STYLE_COLOR,
                            IDC_FELT_ALPHA, uiControls.feltParams.fillAlpha, IDC_STATIC_BASE + 510,
                            IDC_FELT_STYLE_FILLED, uiControls.feltParams.isFilled,
                            yPos);

            yPos += 6;
            createLabel(L"Blue H min:", yPos, IDC_STATIC_BASE + 100);
            createTrackbar(IDC_FELT_BLUE_HMIN, yPos, 0, 180, uiControls.feltParams.blueHMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 101);
            yPos += lineHeight;
            createLabel(L"Blue H max:", yPos, IDC_STATIC_BASE + 102);
            createTrackbar(IDC_FELT_BLUE_HMAX, yPos, 0, 180, uiControls.feltParams.blueHMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 103);
            yPos += lineHeight;
            createLabel(L"Blue S min:", yPos, IDC_STATIC_BASE + 104);
            createTrackbar(IDC_FELT_BLUE_SMIN, yPos, 0, 255, uiControls.feltParams.blueSMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 105);
            yPos += lineHeight;
            createLabel(L"Blue V min:", yPos, IDC_STATIC_BASE + 106);
            createTrackbar(IDC_FELT_BLUE_VMIN, yPos, 0, 255, uiControls.feltParams.blueVMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 107);
            yPos += lineHeight;

            yPos += 6;
            createLabel(L"Green H min:", yPos, IDC_STATIC_BASE + 108);
            createTrackbar(IDC_FELT_GREEN_HMIN, yPos, 0, 180, uiControls.feltParams.greenHMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 109);
            yPos += lineHeight;
            createLabel(L"Green H max:", yPos, IDC_STATIC_BASE + 110);
            createTrackbar(IDC_FELT_GREEN_HMAX, yPos, 0, 180, uiControls.feltParams.greenHMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 111);
            yPos += lineHeight;
            createLabel(L"Green S min:", yPos, IDC_STATIC_BASE + 112);
            createTrackbar(IDC_FELT_GREEN_SMIN, yPos, 0, 255, uiControls.feltParams.greenSMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 113);
            yPos += lineHeight;
            createLabel(L"Green V min:", yPos, IDC_STATIC_BASE + 114);
            createTrackbar(IDC_FELT_GREEN_VMIN, yPos, 0, 255, uiControls.feltParams.greenVMin);
            createValueLabel(yPos, IDC_STATIC_BASE + 115);
            yPos += lineHeight;

            yPos += 6;
            createLabel(L"Outline:", yPos, IDC_STATIC_BASE + 118);
            createTrackbar(IDC_FELT_THICKNESS, yPos, 1, 10, uiControls.feltParams.outlineThicknessPx);
            createValueLabel(yPos, IDC_STATIC_BASE + 119);
            yPos += lineHeight + gap;
        }

        // RAILS group
        {
            addFeatureGroup(L"Rails",
                            IDC_DEBUG_OVERLAY_RAIL_CB, uiControls.showRail,
                            IDC_RAIL_STYLE_COLOR,
                            IDC_RAIL_ALPHA, uiControls.railParams.fillAlpha, IDC_STATIC_BASE + 520,
                            IDC_RAIL_STYLE_FILLED, uiControls.railParams.isFilled,
                            yPos);

            yPos += 6;
            createLabel(L"Black V Max:", yPos, IDC_STATIC_BASE + 10);
            createTrackbar(IDC_RAIL_BLACK_VMAX, yPos, 0, 255, uiControls.railParams.blackVMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 11);
            yPos += lineHeight;

            createLabel(L"Brown H Max:", yPos, IDC_STATIC_BASE + 12);
            createTrackbar(IDC_RAIL_BROWN_HMAX, yPos, 0, 180, uiControls.railParams.brownHMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 13);
            yPos += lineHeight;

            createLabel(L"Brown S Max:", yPos, IDC_STATIC_BASE + 14);
            createTrackbar(IDC_RAIL_BROWN_SMAX, yPos, 0, 255, uiControls.railParams.brownSMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 15);
            yPos += lineHeight;

            createLabel(L"Brown V Max:", yPos, IDC_STATIC_BASE + 16);
            createTrackbar(IDC_RAIL_BROWN_VMAX, yPos, 0, 255, uiControls.railParams.brownVMax);
            createValueLabel(yPos, IDC_STATIC_BASE + 17);
            yPos += lineHeight;

            yPos += 6;
            createLabel(L"Outline:", yPos, IDC_STATIC_BASE + 122);
            createTrackbar(IDC_RAIL_THICKNESS, yPos, 1, 10, uiControls.railParams.outlineThicknessPx);
            createValueLabel(yPos, IDC_STATIC_BASE + 123);
            yPos += lineHeight + gap;
        }

        // DIAMONDS group - Simplified Color Picker UI
        {
            addFeatureGroup(L"Diamonds",
                            IDC_DEBUG_OVERLAY_DIAMONDS_CB, uiControls.showDiamonds,
                            IDC_DIAMONDS_STYLE_COLOR,
                            IDC_DIAMONDS_ALPHA, uiControls.diamondParams.alpha, IDC_STATIC_BASE + 530,
                            IDC_DIAMONDS_STYLE_FILLED, uiControls.diamondParams.isFilled,
                            yPos);

            yPos += 12;
            
            // Header
            {
                HWND hHeader = CreateWindowW(L"STATIC", L"Color Picker", WS_VISIBLE | WS_CHILD | SS_LEFT,
                                            xPos, yPos - g_sidebarScrollPos, usableWidth, 20, g_sidebarPanel, NULL, NULL, NULL);
                applyFont(hHeader, true);
            }
            yPos += lineHeight + 6;

            // Color picker button
            {
                const wchar_t* btnText = g_colorPickerActive ? L"Cancel (Click to Pick)" : L"Pick Diamond Color";
                HWND hColorPicker = CreateWindowW(L"BUTTON", btnText, WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                                 xPos, yPos - g_sidebarScrollPos, usableWidth, rowH + 5, g_sidebarPanel,
                                                 (HMENU)(INT_PTR)IDC_DIAMOND_COLOR_PICKER, NULL, NULL);
                applyFont(hColorPicker, true);
            }
            yPos += lineHeight + 8;

            // Color swatch (visual color display)
            {
                // Label
                createLabel(L"Picked Color:", yPos, IDC_STATIC_BASE + 200);
                yPos += lineHeight;
                
                // Color swatch rectangle
                cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
                COLORREF swatchColor = RGB(bgr[2], bgr[1], bgr[0]); // BGR to RGB
                
                // IMPORTANT:
                // - Do NOT use SS_WHITERECT / SS_BLACKRECT here: those styles cause the STATIC control
                //   to paint a fixed-color rectangle (white/black) in its own WM_PAINT, which prevents
                //   our WM_CTLCOLORSTATIC brush from showing up.
                // - We rely on WM_CTLCOLORSTATIC (handled in SidebarPanelProc) to provide a solid brush.
                HWND hSwatch = CreateWindowW(L"STATIC", L"", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_NOTIFY,
                                            xPos + 10, yPos - g_sidebarScrollPos, usableWidth - 20, 40, 
                                            g_sidebarPanel, (HMENU)(INT_PTR)IDC_COLOR_PICKER_SWATCH, NULL, NULL);
                if (hSwatch) {
                    // Keep the swatch brush in sync even when the sidebar is rebuilt.
                    setColorPickerSwatchBrush(swatchColor);
                    InvalidateRect(hSwatch, NULL, TRUE);
                }
            }
            yPos += 45;

            // Sensitivity slider (tolerance for how far from the picked color we accept)
            {
                // IDs chosen in the IDC_STATIC_BASE + 200+ range to keep them local to the color picker UI.
                createLabel(L"Sensitivity:", yPos, IDC_STATIC_BASE + 205);
                createTrackbar(IDC_DIAMOND_COLOR_SENSITIVITY, yPos, 0, 100, uiControls.diamondParams.colorSensitivity);
                createValueLabel(yPos, IDC_STATIC_BASE + 206);
                yPos += lineHeight + 6;
            }

            // Color info display (compact)
            {
                wchar_t bgrText[64] = L"BGR: --";
                wchar_t hsvText[64] = L"HSV: --";
                wchar_t rangeText[128] = L"Range: --";
                
                // Color filtering is always enabled; show current values unconditionally.
                cv::Vec3b bgr = uiControls.diamondParams.pickedBGR;
                swprintf_s(bgrText, L"BGR: (%d, %d, %d)",
                          (int)bgr[2], (int)bgr[1], (int)bgr[0]);
                swprintf_s(hsvText, L"HSV: (%d, %d, %d)",
                          (uiControls.diamondParams.colorHMin + uiControls.diamondParams.colorHMax) / 2,
                          (uiControls.diamondParams.colorSMin + uiControls.diamondParams.colorSMax) / 2,
                          (uiControls.diamondParams.colorVMin + uiControls.diamondParams.colorVMax) / 2);
                swprintf_s(rangeText, L"Range: H[%d-%d] S[%d-%d] V[%d-%d]",
                          uiControls.diamondParams.colorHMin, uiControls.diamondParams.colorHMax,
                          uiControls.diamondParams.colorSMin, uiControls.diamondParams.colorSMax,
                          uiControls.diamondParams.colorVMin, uiControls.diamondParams.colorVMax);
                
                createLabel(bgrText, yPos, IDC_COLOR_PICKER_BGR);
                yPos += lineHeight;
                createLabel(hsvText, yPos, IDC_COLOR_PICKER_HSV);
                yPos += lineHeight;
                createLabel(rangeText, yPos, IDC_COLOR_PICKER_RANGE_H);
                yPos += lineHeight;
            }

            yPos += lineHeight + gap;
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
            SendMessageW(hCombo, CB_SETITEMDATA, 0, (LPARAM)-1);
            int selectedIndex = 0;
            int idx = 1;
            for (int cam : g_availableCameras) {
                std::wstring label = L"Camera " + std::to_wstring(cam);
                SendMessageW(hCombo, CB_ADDSTRING, 0, (LPARAM)label.c_str());
                SendMessageW(hCombo, CB_SETITEMDATA, idx, (LPARAM)cam);
                if (!uiControls.useTestImage && uiControls.selectedCamera == cam) {
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
    HWND hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_THRESH1);
    if (hTrackbar) {
        int thresholdVal = (uiControls.diamondParams.min_threshold > 0) ? uiControls.diamondParams.min_threshold : uiControls.diamondParams.threshold1;
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, thresholdVal);
    }
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_THRESH2);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.threshold2);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_MINAREA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.minArea);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_MAXAREA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.maxArea);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_CIRCULARITY);
    if (hTrackbar) {
        int circularityVal = static_cast<int>(uiControls.diamondParams.min_circularity * 100.0f);
        SendMessage(hTrackbar, TBM_SETPOS, TRUE, circularityVal);
    }

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_MORPH_KERNEL);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.morph_kernel_size);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMOND_COLOR_SENSITIVITY);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.colorSensitivity);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_BLACK_VMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.blackVMax);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_BROWN_HMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.brownHMax);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_BROWN_SMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.brownSMax);
    
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_BROWN_VMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.brownVMax);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMONDS_RADIUS);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.radiusPx);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMONDS_THICKNESS);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.outlineThicknessPx);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_DIAMONDS_ALPHA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.diamondParams.alpha);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_BLUE_HMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.blueHMin);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_BLUE_HMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.blueHMax);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_BLUE_SMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.blueSMin);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_BLUE_VMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.blueVMin);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_GREEN_HMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.greenHMin);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_GREEN_HMAX);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.greenHMax);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_GREEN_SMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.greenSMin);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_GREEN_VMIN);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.greenVMin);

    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_ALPHA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.fillAlpha);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_FELT_THICKNESS);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.feltParams.outlineThicknessPx);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_ALPHA);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.fillAlpha);
    hTrackbar = GetDlgItem(g_sidebarPanel, IDC_RAIL_THICKNESS);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.railParams.outlineThicknessPx);

            hTrackbar = GetDlgItem(g_sidebarPanel, IDC_SMOOTHING);
    if (hTrackbar) SendMessage(hTrackbar, TBM_SETPOS, TRUE, uiControls.smoothingPercent);
    
    // Update value labels
    wchar_t buffer[32];
    HWND hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 2);
    if (hLabel) {
        int thresholdVal = (uiControls.diamondParams.min_threshold > 0) ? uiControls.diamondParams.min_threshold : uiControls.diamondParams.threshold1;
        swprintf_s(buffer, L"%d", thresholdVal);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 6);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.diamondParams.minArea);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 8);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.diamondParams.maxArea);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 19);
    if (hLabel) {
        swprintf_s(buffer, L"%.2f", uiControls.diamondParams.min_circularity);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 21);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.diamondParams.morph_kernel_size);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 23);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.diamondParams.adaptive_thresh_C);
        SetWindowTextW(hLabel, buffer);
    }
    // Color picker sensitivity value label (0..100)
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 206);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.diamondParams.colorSensitivity);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 11);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.railParams.blackVMax);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 13);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.railParams.brownHMax);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 15);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.railParams.brownSMax);
        SetWindowTextW(hLabel, buffer);
    }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 17);
    if (hLabel) {
        swprintf_s(buffer, L"%d", uiControls.railParams.brownVMax);
        SetWindowTextW(hLabel, buffer);
    }

    // Diamond style values
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 91);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.radiusPx); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 93);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }

    // Felt param values
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 101);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.blueHMin); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 103);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.blueHMax); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 105);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.blueSMin); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 107);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.blueVMin); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 109);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.greenHMin); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 111);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.greenHMax); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 113);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.greenSMin); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 115);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.greenVMin); SetWindowTextW(hLabel, buffer); }

    // Alpha/thickness values
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 401);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.smoothingPercent); SetWindowTextW(hLabel, buffer); }

    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 117);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 119);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 121);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.railParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 123);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.railParams.outlineThicknessPx); SetWindowTextW(hLabel, buffer); }

    // Inline alpha value labels (next to Color buttons)
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 510);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.feltParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 520);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.railParams.fillAlpha); SetWindowTextW(hLabel, buffer); }
    hLabel = GetDlgItem(g_sidebarPanel, IDC_STATIC_BASE + 530);
    if (hLabel) { swprintf_s(buffer, L"%d", uiControls.diamondParams.alpha); SetWindowTextW(hLabel, buffer); }
}

// Handle sidebar button clicks
void handleSidebarButton(int buttonId) {
    if (buttonId == IDC_SIDEBAR_COLLAPSE) {
        uiControls.sidebarCollapsed = !uiControls.sidebarCollapsed;
    }
    else if (buttonId == IDC_SIDEBAR_DIAMONDS) {
        uiControls.sidebarContext = SidebarContext::Diamonds;
    }
    else if (buttonId == IDC_SIDEBAR_FELT) {
        uiControls.sidebarContext = SidebarContext::Felt;
    }
    else if (buttonId == IDC_SIDEBAR_RAIL) {
        uiControls.sidebarContext = SidebarContext::Rail;
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
        case IDC_DIAMOND_THRESH1:
            uiControls.diamondParams.threshold1 = value;
            uiControls.diamondParams.min_threshold = value;  // Keep in sync
            break;
        case IDC_DIAMOND_THRESH2:
            uiControls.diamondParams.threshold2 = value;
            break;
        case IDC_DIAMOND_MINAREA:
            uiControls.diamondParams.minArea = value;
            break;
        case IDC_DIAMOND_MAXAREA:
            uiControls.diamondParams.maxArea = value;
            break;
        case IDC_DIAMOND_CIRCULARITY:
            // Scale from 0-100 to 0.0-1.0
            uiControls.diamondParams.min_circularity = static_cast<float>(value) / 100.0f;
            break;
        case IDC_DIAMOND_MORPH_KERNEL:
            // Ensure odd number for morphological kernel
            uiControls.diamondParams.morph_kernel_size = (value % 2 == 0) ? (value + 1) : value;
            uiControls.diamondParams.morph_kernel_size = std::clamp(uiControls.diamondParams.morph_kernel_size, 5, 31);
            break;
        case IDC_DIAMOND_ADAPTIVE_C:
            uiControls.diamondParams.adaptive_thresh_C = value;
            break;
        case IDC_DIAMOND_COLOR_SENSITIVITY:
            uiControls.diamondParams.colorSensitivity = std::clamp(value, 0, 100);
            // If a color has already been picked, update the HSV range immediately so the filter reacts live.
            applyColorSensitivityToRangesFromPickedHSV();
            updateColorPickerLabels();
            break;
        case IDC_RAIL_BLACK_VMAX:
            uiControls.railParams.blackVMax = value;
            break;
        case IDC_RAIL_BROWN_HMAX:
            uiControls.railParams.brownHMax = value;
            break;
        case IDC_RAIL_BROWN_SMAX:
            uiControls.railParams.brownSMax = value;
            break;
        case IDC_RAIL_BROWN_VMAX:
            uiControls.railParams.brownVMax = value;
            break;
        case IDC_DIAMONDS_RADIUS:
            uiControls.diamondParams.radiusPx = value;
            break;
        case IDC_DIAMONDS_THICKNESS:
            uiControls.diamondParams.outlineThicknessPx = value;
            break;
        case IDC_DIAMONDS_ALPHA:
            uiControls.diamondParams.alpha = value;
            break;
        case IDC_FELT_BLUE_HMIN:
            uiControls.feltParams.blueHMin = value;
            break;
        case IDC_FELT_BLUE_HMAX:
            uiControls.feltParams.blueHMax = value;
            break;
        case IDC_FELT_BLUE_SMIN:
            uiControls.feltParams.blueSMin = value;
            break;
        case IDC_FELT_BLUE_VMIN:
            uiControls.feltParams.blueVMin = value;
            break;
        case IDC_FELT_GREEN_HMIN:
            uiControls.feltParams.greenHMin = value;
            break;
        case IDC_FELT_GREEN_HMAX:
            uiControls.feltParams.greenHMax = value;
            break;
        case IDC_FELT_GREEN_SMIN:
            uiControls.feltParams.greenSMin = value;
            break;
        case IDC_FELT_GREEN_VMIN:
            uiControls.feltParams.greenVMin = value;
            break;
        case IDC_FELT_ALPHA:
            uiControls.feltParams.fillAlpha = value;
            break;
        case IDC_FELT_THICKNESS:
            uiControls.feltParams.outlineThicknessPx = value;
            break;
        case IDC_RAIL_ALPHA:
            uiControls.railParams.fillAlpha = value;
            break;
        case IDC_RAIL_THICKNESS:
            uiControls.railParams.outlineThicknessPx = value;
            break;
        case IDC_SMOOTHING:
            uiControls.smoothingPercent = value;
            break;
        default:
            // Handle felt parameters (dynamic IDs)
            if (trackbarId == IDC_STATIC_BASE + 21) uiControls.feltParams.blueHMin = value;
            else if (trackbarId == IDC_STATIC_BASE + 24) uiControls.feltParams.blueHMax = value;
            else if (trackbarId == IDC_STATIC_BASE + 27) uiControls.feltParams.greenHMin = value;
            else if (trackbarId == IDC_STATIC_BASE + 30) uiControls.feltParams.greenHMax = value;
            break;
    }
    updateSidebarControls();
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
        createSidebarControls(g_hwnd);
    }
    else if (menuId == IDM_NAV_DISPLAY) {
        uiControls.showSidebar = true;
        uiControls.sidebarPage = SidebarPage::Display;
        updateOverlayMenu();
        layoutChildren(g_hwnd);
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

