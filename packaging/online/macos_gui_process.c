#include "macos_gui_process.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

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
