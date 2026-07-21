/* Compiled launcher for the run-from-source .app bundle.
 * A Mach-O CFBundleExecutable (not a shell script) is required for macOS TCC to
 * attribute camera/Bluetooth permission to the app and show the prompt.
 * execv replaces this image with the venv Python in the SAME process, so TCC's
 * grant to this signed binary carries to the running app. Paths are baked at
 * build time (dedicated single-purpose rig). */
#include <unistd.h>
#include <stdlib.h>

int main(void) {
    const char *repo = "/Users/joe/Documents/GitHub/_CaulfieldEngineering/BilliardsTrainer";
    const char *py   = "/Users/joe/Documents/GitHub/_CaulfieldEngineering/BilliardsTrainer/.venv/bin/python3";
    chdir(repo);
    char *const argv[] = { (char*)py, "-m", "billiards_trainer", NULL };
    execv(py, argv);
    return 1; /* only reached if execv fails */
}
