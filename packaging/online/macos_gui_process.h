#ifndef SPACR_MACOS_GUI_PROCESS_H
#define SPACR_MACOS_GUI_PROCESS_H
#include <sys/types.h>
#include <stddef.h>
struct spacr_install_process { pid_t pid; int output; };
int spacr_install_spawn(char *const argv[], char *const envp[], struct spacr_install_process *process);
int spacr_install_cancel(pid_t pid);
int spacr_install_wait(pid_t pid);
int spacr_uninstall_tree(const char *path, dev_t device, ino_t inode, uid_t owner,
                         const char *const kept[], size_t kept_count, int apply);
#endif
