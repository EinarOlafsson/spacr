#include "macos_gui_process.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int spacr_install_spawn(char *const argv[], char *const envp[], struct spacr_install_process *process) {
    int descriptors[2];
    process->pid = 0;
    process->output = -1;
    if (pipe(descriptors) != 0) return errno;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) goto close_pipe;
    error = posix_spawnattr_init(&attributes);
    if (error) goto destroy_actions;
    error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (!error) error = posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    if (!error) error = posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO);
    if (!error) error = posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    if (!error) error = posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    if (!error) error = posix_spawnattr_setpgroup(&attributes, 0);
    if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    if (!error) error = posix_spawn(&process->pid, argv[0], &actions, &attributes, argv, envp);
    posix_spawnattr_destroy(&attributes);
destroy_actions:
    posix_spawn_file_actions_destroy(&actions);
close_pipe:
    close(descriptors[1]);
    if (error) close(descriptors[0]);
    else process->output = descriptors[0];
    return error;
}

int spacr_install_cancel(pid_t pid) {
    if (pid <= 0) { errno = EINVAL; return -1; }
    return kill(-pid, SIGTERM);
}

int spacr_install_wait(pid_t pid) {
    int status;
    pid_t observed;
    do { observed = waitpid(pid, &status, 0); } while (observed < 0 && errno == EINTR);
    if (observed < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

static int spacr_path_inside(const char *path, const char *parent) {
    size_t length = strlen(parent);
    return !strncmp(path, parent, length) && (!path[length] || path[length] == '/');
}

static int spacr_remove_entry(int parent, const char *name, const char *path,
                            const struct stat *expected, uid_t owner,
                            const char *const kept[], size_t count, int apply) {
    for (size_t index = 0; index < count; ++index)
        if (spacr_path_inside(path, kept[index])) return 0;
    struct stat before, opened, after;
    if (fstatat(parent, name, &before, AT_SYMLINK_NOFOLLOW)) return errno;
    if (expected && (before.st_dev != expected->st_dev || before.st_ino != expected->st_ino)) return ESTALE;
    if (S_ISDIR(before.st_mode)) {
        if (before.st_uid != owner || (before.st_mode & 0022)) return EPERM;
        int descriptor = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
        if (descriptor < 0) return errno;
        if (fstat(descriptor, &opened) || opened.st_dev != before.st_dev || opened.st_ino != before.st_ino) {
            close(descriptor);
            return ESTALE;
        }
        DIR *directory = fdopendir(descriptor);
        if (!directory) { int error = errno; close(descriptor); return error; }
        int error = 0;
        struct dirent *entry;
        for (;;) {
            errno = 0;
            entry = readdir(directory);
            if (!entry) { error = errno; break; }
            if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
            size_t length = strlen(path) + strlen(entry->d_name) + 2;
            char *child = malloc(length);
            if (!child) { error = ENOMEM; break; }
            snprintf(child, length, "%s/%s", path, entry->d_name);
            error = spacr_remove_entry(descriptor, entry->d_name, child, NULL, owner, kept, count, apply);
            free(child);
            if (error) break;
        }
        closedir(directory);
        if (error || !apply) return error;
        for (size_t index = 0; index < count; ++index)
            if (spacr_path_inside(kept[index], path)) return 0;
    } else if (!S_ISREG(before.st_mode) && !S_ISLNK(before.st_mode)) return EINVAL;
    if (!apply) return 0;
    if (fstatat(parent, name, &after, AT_SYMLINK_NOFOLLOW) || after.st_dev != before.st_dev
            || after.st_ino != before.st_ino || (after.st_mode & S_IFMT) != (before.st_mode & S_IFMT)) return ESTALE;
    return unlinkat(parent, name, S_ISDIR(before.st_mode) ? AT_REMOVEDIR : 0) ? errno : 0;
}

int spacr_uninstall_tree(const char *path, dev_t device, ino_t inode, uid_t owner,
                         const char *const kept[], size_t kept_count, int apply) {
    if (!path || path[0] != '/' || !path[1] || path[strlen(path) - 1] == '/') return EINVAL;
    char *folder = strdup(path);
    if (!folder) return ENOMEM;
    char *name = strrchr(folder, '/');
    *name++ = '\0';
    int parent = open(*folder ? folder : "/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
    if (parent < 0) { int error = errno; free(folder); return error; }
    struct stat expected = {.st_dev = device, .st_ino = inode};
    int error = spacr_remove_entry(parent, name, path, &expected, owner, kept, kept_count, apply);
    close(parent);
    free(folder);
    return error;
}
