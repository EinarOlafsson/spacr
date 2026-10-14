/* Small native entry point: scientific dependencies stay in the private venv. */
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <limits.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef SPACR_PACKAGE_VERSION
#define SPACR_PACKAGE_VERSION ""
#endif
#ifndef SPACR_SUPPORT_DIR
#define SPACR_SUPPORT_DIR "/Library/Application Support/spaCR"
#endif
#ifndef SPACR_GUI_EXECUTABLE
#define SPACR_GUI_EXECUTABLE "/Applications/spaCR.app/Contents/MacOS/install-spacr-gui"
#endif

static int runtime_verified(const char *python) {
    if (!SPACR_PACKAGE_VERSION[0]) return 1;
    pid_t child = fork();
    if (child < 0) return 0;
    if (child == 0) {
        int silent = open("/dev/null", O_WRONLY);
        if (silent >= 0) {
            dup2(silent, STDOUT_FILENO);
            dup2(silent, STDERR_FILENO);
            close(silent);
        }
        execl(python, python, "-I", SPACR_SUPPORT_DIR "/verify_runtime.py", "verify-runtime",
              "--python", python, "--version", SPACR_PACKAGE_VERSION, "--minimum", (char *)NULL);
        _exit(127);
    }
    int status;
    pid_t observed;
    do { observed = waitpid(child, &status, 0); } while (observed < 0 && errno == EINTR);
    return observed == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main(int argc, char **argv) {
    const char *home = getenv("HOME");
    if (!home || !*home) {
        struct passwd *user = getpwuid(getuid());
        home = user ? user->pw_dir : NULL;
    }
    if (!home || !*home) {
        fputs("spaCR could not locate your home directory.\n", stderr);
        return 1;
    }
    char python[PATH_MAX];
    int length = snprintf(python, sizeof(python),
                          "%s/Library/Application Support/spaCR/venv/bin/python", home);
    if (length < 0 || (size_t)length >= sizeof(python)) {
        fputs("spaCR runtime path is too long.\n", stderr);
        return 1;
    }
    if (access(python, X_OK) != 0 || !runtime_verified(python)) {
        const char *installer = SPACR_GUI_EXECUTABLE;
        char **setup_args = calloc((size_t)argc + 1, sizeof(char *));
        if (!setup_args) { perror("spaCR installer allocation"); return 1; }
        setup_args[0] = (char *)installer;
        for (int index = 1; index < argc; ++index) setup_args[index] = argv[index];
        execv(installer, setup_args);
        free(setup_args);
        perror("spaCR could not start the first-run installer");
        return 1;
    }
    char **args = calloc((size_t)argc + 4, sizeof(char *));
    if (!args) {
        perror("spaCR launcher allocation");
        return 1;
    }
    args[0] = python;
    args[1] = "-I";
    args[2] = "-m";
    args[3] = "spacr.qt";
    for (int index = 1; index < argc; ++index)
        args[index + 3] = argv[index];
    execv(python, args);
    perror("spaCR could not launch its private Python runtime");
    free(args);
    return 1;
}
