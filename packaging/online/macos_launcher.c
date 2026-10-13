/* Small native entry point: scientific dependencies stay in the private venv. */
#include <errno.h>
#include <limits.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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
    if (access(python, X_OK) != 0) {
        execl("/usr/bin/osascript", "osascript", "-e",
              "tell application \"Terminal\"\nactivate\n"
              "do script quoted form of \"/Library/Application Support/spaCR/install-for-user.sh\"\n"
              "end tell", (char *)NULL);
        perror("spaCR could not start the first-run installer");
        return 1;
    }
    char **args = calloc((size_t)argc + 3, sizeof(char *));
    if (!args) {
        perror("spaCR launcher allocation");
        return 1;
    }
    args[0] = python;
    args[1] = "-m";
    args[2] = "spacr.qt";
    for (int index = 1; index < argc; ++index)
        args[index + 2] = argv[index];
    execv(python, args);
    perror("spaCR could not launch its private Python runtime");
    free(args);
    return 1;
}
