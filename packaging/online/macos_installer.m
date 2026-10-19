#import <Cocoa/Cocoa.h>
#include "macos_gui_process.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <libproc.h>
#include <limits.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>

static BOOL uninstallInside(NSString *path, NSString *parent) {
    return [path isEqualToString:parent] || [path hasPrefix:[parent stringByAppendingString:@"/"]];
}

static NSString *uninstallMAC(NSData *secret, NSDictionary *payload) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
        options:NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes error:NULL];
    if (!data) return nil;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CCHmac(kCCHmacAlgSHA256, secret.bytes, secret.length, data.bytes, data.length, digest);
    NSMutableString *text = [NSMutableString string];
    for (NSUInteger index = 0; index < sizeof(digest); ++index) [text appendFormat:@"%02x", digest[index]];
    return text;
}

static BOOL uninstallSend(int channel, NSData *secret, NSString *phase, int sequence) {
    NSDictionary *payload = @{@"details": @{}, @"direction": @"installer", @"phase": phase, @"sequence": @(sequence)};
    NSString *mac = uninstallMAC(secret, payload);
    if (!mac) return NO;
    NSMutableData *data = [[NSJSONSerialization dataWithJSONObject:@{@"payload": payload, @"mac": mac}
        options:0 error:NULL] mutableCopy];
    [data appendBytes:"\n" length:1];
    size_t written = 0;
    while (written < data.length) {
        ssize_t count = send(channel, (const char *)data.bytes + written, data.length - written, 0);
        if (count <= 0) { if (errno == EINTR) continue; return NO; }
        written += (size_t)count;
    }
    return YES;
}

static BOOL uninstallApproved(int channel, NSData *secret) {
    NSMutableData *data = [NSMutableData data];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:600];
    while (deadline.timeIntervalSinceNow > 0 && data.length < 8192) {
        char byte;
        ssize_t count = recv(channel, &byte, 1, 0);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count != 1) return NO;
        if (byte != '\n') { [data appendBytes:&byte length:1]; continue; }
        id message = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
        if (![message isKindOfClass:NSDictionary.class]) return NO;
        id payload = message[@"payload"];
        if (![payload isKindOfClass:NSDictionary.class] || [payload count] != 4
                || ![payload[@"details"] isEqual:@{}] || ![payload[@"direction"] isEqual:@"desktop"]
                || ![payload[@"sequence"] isEqual:@1] || ![payload[@"phase"] isEqual:@"activate"]) return NO;
        NSString *expected = uninstallMAC(secret, payload);
        id actual = message[@"mac"];
        if (![actual isKindOfClass:NSString.class] || [actual length] != expected.length) return NO;
        unsigned difference = 0;
        for (NSUInteger index = 0; index < expected.length; ++index)
            difference |= [actual characterAtIndex:index] ^ [expected characterAtIndex:index];
        return difference == 0;
    }
    return NO;
}

static int uninstallChannel(const char *descriptorPath, pid_t pid, const char *desktop, NSData **secret) {
    struct proc_bsdinfo process;
    char observed[PROC_PIDPATHINFO_MAXSIZE], expected[PATH_MAX];
    if (pid <= 1 || proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &process, sizeof(process)) != (int)sizeof(process)
            || proc_pidpath(pid, observed, sizeof(observed)) <= 0 || !realpath(desktop, expected)
            || strcmp(observed, expected)) return -1;
    int descriptor = open(descriptorPath, O_RDONLY | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) return -1;
    struct stat details;
    char text[130] = {0};
    ssize_t count = -1;
    if (!fstat(descriptor, &details) && S_ISREG(details.st_mode) && details.st_uid == process.pbi_uid
            && !(details.st_mode & 0077) && details.st_size <= 128) count = read(descriptor, text, 129);
    close(descriptor);
    if (count < 67 || count > 70) return -1;
    char *separator = strchr(text, ':');
    if (!separator || strlen(separator + 1) != 64) return -1;
    *separator = '\0';
    char *end;
    long port = strtol(text, &end, 10);
    if (*end || port <= 0 || port > 65535) return -1;
    unsigned char bytes[32];
    for (size_t index = 0; index < 32; ++index) {
        char pair[3] = {separator[1 + index * 2], separator[2 + index * 2], 0};
        if (strspn(pair, "0123456789abcdef") != 2) return -1;
        bytes[index] = (unsigned char)strtoul(pair, NULL, 16);
    }
    *secret = [NSData dataWithBytes:bytes length:sizeof(bytes)];
    int channel = socket(AF_INET, SOCK_STREAM, 0);
    if (channel < 0) return -1;
    struct timeval timeout = {.tv_sec = 1};
    int noSignal = 1;
    setsockopt(channel, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(channel, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(channel, SOL_SOCKET, SO_NOSIGPIPE, &noSignal, sizeof(noSignal));
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(channel, (struct sockaddr *)&address, sizeof(address))) { close(channel); return -1; }
    return channel;
}

static int uninstallTarget(NSDictionary *target, BOOL apply) {
    NSString *path = target[@"path"];
    NSArray *identity = target[@"identity"], *kept = target[@"keep"];
    const char **paths = calloc(kept.count + 1, sizeof(char *));
    if (!paths) return ENOMEM;
    for (NSUInteger index = 0; index < kept.count; ++index) paths[index] = [kept[index] fileSystemRepresentation];
    int result = spacr_uninstall_tree(path.fileSystemRepresentation, [identity[0] unsignedLongLongValue],
        [identity[1] unsignedLongLongValue], 0, paths, kept.count, apply);
    free(paths);
    return result;
}

static int uninstallNative(int argc, const char *argv[]) {
    if (argc != 6 || geteuid() != 0 || strlen(argv[5]) > 65536) return EINVAL;
    char *end;
    long parsedPID = strtol(argv[3], &end, 10);
    if (*end || parsedPID <= 1 || parsedPID > INT_MAX) return EINVAL;
    NSData *data = [[NSString stringWithUTF8String:argv[5]] dataUsingEncoding:NSUTF8StringEncoding];
    id request = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
    if (![request isKindOfClass:NSDictionary.class]) return EINVAL;
    id targets = request[@"targets"];
    if (![targets isKindOfClass:NSArray.class] || [targets count] < 1 || [targets count] > 2) return EINVAL;
    NSString *app = @"/Applications/spaCR.app";
    NSString *support = @"/Library/Application Support/spaCR";
    char selfExecutable[PROC_PIDPATHINFO_MAXSIZE];
    if (proc_pidpath(getpid(), selfExecutable, sizeof(selfExecutable)) <= 0) return EPERM;
    NSString *selfPath = [[NSString stringWithUTF8String:selfExecutable] stringByResolvingSymlinksInPath];
    if (![selfPath isEqualToString:[app stringByAppendingPathComponent:@"Contents/MacOS/install-spacr-gui"]]) return EPERM;
    NSData *infoData = [NSData dataWithContentsOfFile:[app stringByAppendingPathComponent:@"Contents/Info.plist"]];
    id info = infoData ? [NSPropertyListSerialization propertyListWithData:infoData options:NSPropertyListImmutable
        format:NULL error:NULL] : nil;
    if (![info isKindOfClass:NSDictionary.class]) return EINVAL;
    if (![info[@"CFBundleIdentifier"] isEqualToString:@"com.einarolafsson.spacr"]
            || ![info[@"CFBundleExecutable"] isEqualToString:@"spaCR"]) return EINVAL;
    NSMutableSet *seen = [NSMutableSet set];
    NSMutableArray *validated = [NSMutableArray array];
    NSSet *supportFiles = [NSSet setWithArray:@[@"install-online.sh", @"install-for-user.sh", @"installer_messages.sh",
        @"uninstall-spacr.sh", @"verify_runtime.py"]];
    for (id target in targets) {
        if (![target isKindOfClass:NSDictionary.class]) return EINVAL;
        id path = target[@"path"], identity = target[@"identity"], keep = target[@"keep"];
        if (!([path isEqual:app] || [path isEqual:support]) || [seen containsObject:path]
                || ![identity isKindOfClass:NSArray.class] || [identity count] != 2
                || ![identity[0] isKindOfClass:NSNumber.class] || ![identity[1] isKindOfClass:NSNumber.class]
                || ![keep isKindOfClass:NSArray.class] || [keep count] > 256) return EINVAL;
        [seen addObject:path];
        NSMutableArray *preserved = [keep mutableCopy];
        for (id item in keep)
            if (![item isKindOfClass:NSString.class] || ![item isEqual:[item stringByStandardizingPath]]
                    || !uninstallInside(item, path)) return EINVAL;
        NSError *error;
        NSArray *children = [NSFileManager.defaultManager contentsOfDirectoryAtPath:path error:&error];
        if (!children) return EIO;
        for (NSString *child in children)
            if (([path isEqual:app] && ![child isEqualToString:@"Contents"])
                    || ([path isEqual:support] && ![supportFiles containsObject:child]))
                [preserved addObject:[path stringByAppendingPathComponent:child]];
        NSDictionary *bounded = @{@"path": path, @"identity": identity, @"keep": preserved};
        int failure = uninstallTarget(bounded, NO);
        if (failure) return failure;
        [validated addObject:bounded];
    }
    if (![seen containsObject:app]) return EINVAL;
    NSString *launcher = @"/usr/local/bin/spacr";
    id removeLauncher = request[@"launcher"];
    if (![removeLauncher isKindOfClass:NSNumber.class]) return EINVAL;
    struct stat launcherIdentity;
    NSString *expectedLink = [app stringByAppendingPathComponent:@"Contents/MacOS/spaCR"];
    if ([removeLauncher boolValue] && (lstat(launcher.fileSystemRepresentation, &launcherIdentity)
            || !S_ISLNK(launcherIdentity.st_mode)
            || ![[NSFileManager.defaultManager destinationOfSymbolicLinkAtPath:launcher error:NULL] isEqual:expectedLink])) return ESTALE;
    NSData *secret;
    int channel = uninstallChannel(argv[2], (pid_t)parsedPID, argv[4], &secret);
    if (channel < 0) return EPERM;
    int result = ECANCELED;
    if (!uninstallSend(channel, secret, @"prepared", 1) || !uninstallApproved(channel, secret)) goto done;
    if (kill((pid_t)parsedPID, 0) == 0 || errno != ESRCH) { result = EBUSY; goto done; }
    for (NSDictionary *target in validated) {
        result = uninstallTarget(target, NO);
        if (result) goto done;
    }
    if ([removeLauncher boolValue]) {
        struct stat current;
        if (lstat(launcher.fileSystemRepresentation, &current) || current.st_dev != launcherIdentity.st_dev
                || current.st_ino != launcherIdentity.st_ino || !S_ISLNK(current.st_mode)
                || ![[NSFileManager.defaultManager destinationOfSymbolicLinkAtPath:launcher error:NULL] isEqual:expectedLink]
                || unlink(launcher.fileSystemRepresentation)) { result = ESTALE; goto done; }
    }
    for (NSDictionary *target in validated) {
        result = uninstallTarget(target, YES);
        if (result) goto done;
    }
    result = uninstallSend(channel, secret, @"complete", 2) ? 0 : EPIPE;
done:
    close(channel);
    return result;
}

static NSDictionary *messages;
static NSString *language;
static NSString *tr(NSString *key) { return messages[key] ?: key; }

@interface SpacrInstaller : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(strong) NSWindow *window;
@property(strong) NSTextView *log;
@property(strong) NSTextField *status;
@property(strong) NSProgressIndicator *progress;
@property(strong) NSButton *install;
@property(strong) NSButton *cancel;
@property(strong) NSButton *uninstall;
@property(strong) NSString *uninstallPreview;
@property(strong) NSString *stagedPython;
@property BOOL removed;
@property(strong) NSMutableArray<NSButton *> *choices;
@property(strong) NSArray<NSString *> *forwardedArguments;
@property pid_t child;
@property BOOL cancelling;
@property BOOL ready;
@end

@implementation SpacrInstaller
- (NSTextField *)label:(NSString *)text frame:(NSRect)frame {
    NSTextField *field = [NSTextField wrappingLabelWithString:text];
    field.frame = frame;
    [self.window.contentView addSubview:field];
    return field;
}
- (NSButton *)button:(NSString *)title frame:(NSRect)frame action:(SEL)action {
    NSButton *button = [NSButton buttonWithTitle:title target:self action:action];
    button.frame = frame;
    [self.window.contentView addSubview:button];
    return button;
}
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 760, 546)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        backing:NSBackingStoreBuffered defer:NO];
    self.window.title = tr(@"installer_title");
    self.window.delegate = self;
    [self.window center];
    [self label:tr(@"gui_explanation") frame:NSMakeRect(24, 484, 712, 44)];
    self.choices = [NSMutableArray array];
    NSArray *keys = @[@"nsis_gpu", @"gui_share_diagnostics", @"gui_report_issues", @"without_wgpu"];
    for (NSUInteger index = 0; index < keys.count; ++index) {
        NSButton *choice = [NSButton checkboxWithTitle:tr(keys[index]) target:nil action:nil];
        choice.frame = NSMakeRect(24, 428 - index * 44, 712, 40);
        [(NSCell *)choice.cell setWraps:YES];
        choice.toolTip = choice.title;
        choice.state = index < 3 ? NSControlStateValueOn : NSControlStateValueOff;
        [self.window.contentView addSubview:choice];
        [self.choices addObject:choice];
    }
    self.status = [self label:tr(@"gui_preserve") frame:NSMakeRect(24, 242, 712, 46)];
    self.progress = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(24, 216, 712, 16)];
    self.progress.indeterminate = YES;
    self.progress.style = NSProgressIndicatorStyleBar;
    [self.window.contentView addSubview:self.progress];
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(24, 72, 712, 132)];
    scroll.hasVerticalScroller = YES;
    self.log = [[NSTextView alloc] initWithFrame:scroll.bounds];
    self.log.editable = NO;
    self.log.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    scroll.documentView = self.log;
    [self.window.contentView addSubview:scroll];
    self.cancel = [self button:tr(@"gui_close") frame:NSMakeRect(500, 20, 110, 36) action:@selector(cancel:)];
    self.install = [self button:tr(@"gui_install") frame:NSMakeRect(620, 20, 116, 36) action:@selector(install:)];
    self.uninstall = [self button:tr(@"gui_uninstall") frame:NSMakeRect(24, 20, 200, 36) action:@selector(uninstall:)];
    self.uninstall.toolTip = tr(@"gui_uninstall_title");
    [self.window makeKeyAndOrderFront:nil];
    if (@available(macOS 14.0, *)) {
        [NSApp activate];
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [NSApp activateIgnoringOtherApps:YES];
#pragma clang diagnostic pop
    }
}
- (void)appendLog:(NSString *)text {
    [self.log.textStorage appendAttributedString:[[NSAttributedString alloc] initWithString:text]];
    NSUInteger length = self.log.textStorage.length;
    if (length > 1024 * 1024) [self.log.textStorage deleteCharactersInRange:NSMakeRange(0, length - 1024 * 1024)];
    [self.log scrollRangeToVisible:NSMakeRange(self.log.textStorage.length, 0)];
}
- (void)setBusy:(BOOL)busy {
    self.install.enabled = !busy && !self.removed;
    self.uninstall.enabled = !busy && !self.removed;
    for (NSButton *choice in self.choices) choice.enabled = !busy;
    self.cancel.title = tr(busy ? @"gui_cancel" : @"gui_close");
    self.cancel.enabled = YES;
    if (busy) [self.progress startAnimation:nil];
    else [self.progress stopAnimation:nil];
}
- (void)finished:(int)code {
    self.child = 0;
    [self setBusy:NO];
    self.ready = code == 0 && !self.cancelling;
    self.status.stringValue = tr(self.ready ? @"ready_opening" : (self.cancelling ? @"gui_cancelled" : @"gui_failed"));
    self.install.title = tr(self.ready ? @"nsis_launch" : @"gui_retry");
    if (self.ready) [self openApplication];
}
- (void)openApplication {
    self.install.enabled = NO;
    NSWorkspaceOpenConfiguration *configuration = [NSWorkspaceOpenConfiguration configuration];
    configuration.arguments = self.forwardedArguments;
    configuration.createsNewApplicationInstance = YES;
    NSURL *app = [NSURL fileURLWithPath:@"/Applications/spaCR.app"];
    [[NSWorkspace sharedWorkspace] openApplicationAtURL:app configuration:configuration
        completionHandler:^(NSRunningApplication *application, NSError *error) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (application && !error) [NSApp terminate:nil];
                else {
                    [self appendLog:[NSString stringWithFormat:@"%@\n", error.localizedDescription ?: tr(@"gui_failed")]];
                    self.status.stringValue = tr(@"gui_failed");
                    self.install.enabled = YES;
                }
            });
        }];
}
- (void)install:(id)sender {
    (void)sender;
    if (self.ready) { [self openApplication]; return; }
    if (self.child > 0) return;
    self.cancelling = NO;
    [self setBusy:YES];
    self.status.stringValue = tr(@"nsis_downloading");
    NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithArray:
        @[@"/bin/sh", @"/Library/Application Support/spaCR/install-for-user.sh", @"--torch-backend",
          self.choices[0].state == NSControlStateValueOn ? @"auto" : @"cpu"]];
    NSArray *flags = @[@"--share-diagnostics", @"--report-issues", @"--without-wgpu"];
    for (NSUInteger index = 0; index < flags.count; ++index)
        if (self.choices[index + 1].state == NSControlStateValueOn) [arguments addObject:flags[index]];
    NSMutableDictionary *environment = [NSProcessInfo.processInfo.environment mutableCopy];
    [environment removeObjectForKey:@"SPACR_USER_INSTALL_ROOT"];
    environment[@"SPACR_NO_RELAUNCH"] = @"1";
    environment[@"SPACR_INSTALL_LANGUAGE"] = language;
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (NSString *key in environment) [entries addObject:[NSString stringWithFormat:@"%@=%@", key, environment[key]]];
    char **argv = calloc(arguments.count + 1, sizeof(char *));
    char **envp = calloc(entries.count + 1, sizeof(char *));
    if (!argv || !envp) { free(argv); free(envp); [self finished:ENOMEM]; return; }
    for (NSUInteger index = 0; index < arguments.count; ++index) argv[index] = (char *)arguments[index].UTF8String;
    for (NSUInteger index = 0; index < entries.count; ++index) envp[index] = (char *)entries[index].UTF8String;
    struct spacr_install_process process;
    int error = spacr_install_spawn(argv, envp, &process);
    free(argv); free(envp);
    if (error) {
        [self appendLog:[NSString stringWithFormat:@"%s\n", strerror(error)]];
        [self finished:error];
        return;
    }
    self.child = process.pid;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        char buffer[8192];
        ssize_t count;
        while ((count = read(process.output, buffer, sizeof(buffer))) != 0) {
            if (count < 0) { if (errno == EINTR) continue; break; }
            NSData *data = [NSData dataWithBytes:buffer length:(NSUInteger)count];
            NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (!text) text = [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding];
            dispatch_sync(dispatch_get_main_queue(), ^{ [self appendLog:text ?: @""]; });
        }
        close(process.output);
        int code = spacr_install_wait(process.pid);
        dispatch_async(dispatch_get_main_queue(), ^{ [self finished:code]; });
    });
}
- (NSString *)helperPython {
    NSString *root = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/spaCR"];
    NSFileManager *files = NSFileManager.defaultManager;
    NSMutableArray<NSString *> *candidates = [NSMutableArray array];
    NSString *pythons = [root stringByAppendingPathComponent:@"python"];
    for (NSString *name in [[files contentsOfDirectoryAtPath:pythons error:NULL] sortedArrayUsingSelector:@selector(compare:)])
        [candidates addObject:[pythons stringByAppendingPathComponent:[name stringByAppendingPathComponent:@"bin/python3"]]];
    [candidates addObject:[root stringByAppendingPathComponent:@"venv/bin/python"]];
    if ([files fileExistsAtPath:@"/Library/Developer/CommandLineTools/usr/bin/python3"]
            || [files fileExistsAtPath:@"/Applications/Xcode.app/Contents/Developer/usr/bin/python3"])
        [candidates addObject:@"/usr/bin/python3"];
    for (NSString *candidate in candidates)
        if ([files isExecutableFileAtPath:candidate]) return candidate;
    return nil;
}
- (NSString *)stageOutsideRuntime:(NSString *)python {
    NSString *root = [[NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/spaCR"]
        stringByResolvingSymlinksInPath];
    NSString *real = [python stringByResolvingSymlinksInPath];
    if (![real hasPrefix:[root stringByAppendingString:@"/"]]) return python;
    NSString *home = [[real stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
    NSString *copy = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"spacr-uninstall-python-%@", NSUUID.UUID.UUIDString]];
    NSError *error = nil;
    if (![NSFileManager.defaultManager copyItemAtPath:home toPath:copy error:&error]) {
        [self appendLog:[NSString stringWithFormat:@"%@\n", error.localizedDescription ?: @""]];
        return nil;
    }
    self.stagedPython = copy;
    return [copy stringByAppendingPathComponent:[@"bin" stringByAppendingPathComponent:real.lastPathComponent]];
}
- (void)removeStagedPython {
    if (self.stagedPython) [NSFileManager.defaultManager removeItemAtPath:self.stagedPython error:NULL];
    self.stagedPython = nil;
}
- (void)runHelper:(NSArray<NSString *> *)arguments completion:(void (^)(int code, NSString *output))completion {
    NSMutableDictionary *environment = [NSProcessInfo.processInfo.environment mutableCopy];
    environment[@"SPACR_INSTALL_LANGUAGE"] = language;
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (NSString *key in environment) [entries addObject:[NSString stringWithFormat:@"%@=%@", key, environment[key]]];
    char **argv = calloc(arguments.count + 1, sizeof(char *));
    char **envp = calloc(entries.count + 1, sizeof(char *));
    if (!argv || !envp) { free(argv); free(envp); completion(ENOMEM, @""); return; }
    for (NSUInteger index = 0; index < arguments.count; ++index) argv[index] = (char *)arguments[index].UTF8String;
    for (NSUInteger index = 0; index < entries.count; ++index) envp[index] = (char *)entries[index].UTF8String;
    struct spacr_install_process process;
    int error = spacr_install_spawn(argv, envp, &process);
    free(argv); free(envp);
    if (error) { completion(error, [NSString stringWithUTF8String:strerror(error)]); return; }
    self.child = process.pid;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSMutableString *collected = [NSMutableString string];
        char buffer[8192];
        ssize_t count;
        while ((count = read(process.output, buffer, sizeof(buffer))) != 0) {
            if (count < 0) { if (errno == EINTR) continue; break; }
            NSString *text = [[NSString alloc] initWithData:[NSData dataWithBytes:buffer length:(NSUInteger)count]
                encoding:NSUTF8StringEncoding] ?: @"";
            [collected appendString:text];
            dispatch_sync(dispatch_get_main_queue(), ^{ [self appendLog:text]; });
        }
        close(process.output);
        int code = spacr_install_wait(process.pid);
        dispatch_async(dispatch_get_main_queue(), ^{ self.child = 0; completion(code, collected); });
    });
}
- (void)uninstall:(id)sender {
    (void)sender;
    if (self.child > 0) return;
    NSString *python = [self helperPython];
    if (python) python = [self stageOutsideRuntime:python];
    NSString *helper = @"/Library/Application Support/spaCR/verify_runtime.py";
    [self setBusy:YES];
    self.status.stringValue = tr(@"gui_uninstall_finding");
    if (!python || ![NSFileManager.defaultManager fileExistsAtPath:helper]) {
        [self appendLog:[tr(@"gui_uninstall_no_python") stringByAppendingString:@"\n"]];
        [self confirmUninstall:nil python:nil];
        return;
    }
    NSString *output = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"spacr-uninstall-%@.json", NSUUID.UUID.UUIDString]];
    [self runHelper:@[python, @"-I", helper, @"uninstall-preview", @"--output", output]
         completion:^(int code, NSString *listing) {
        if (code) { [self uninstallStopped]; return; }
        self.uninstallPreview = output;
        [self confirmUninstall:listing python:python];
    }];
}
- (void)confirmUninstall:(NSString *)listing python:(NSString *)python {
    NSArray *keys = @[@"caches", @"models", @"logs", @"settings", @"history", @"temporary", @"downloads", @"other"];
    NSView *accessory = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 560, 360)];
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, 0, 560, 170)];
    scroll.hasVerticalScroller = YES;
    NSTextView *paths = [[NSTextView alloc] initWithFrame:scroll.bounds];
    paths.editable = NO;
    paths.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    paths.string = listing ?: @"";
    scroll.documentView = paths;
    [accessory addSubview:scroll];
    NSMutableArray<NSButton *> *boxes = [NSMutableArray array];
    for (NSUInteger index = 0; index < keys.count; ++index) {
        NSButton *box = [NSButton checkboxWithTitle:tr([@"gui_category_" stringByAppendingString:keys[index]]) target:nil action:nil];
        box.frame = NSMakeRect((index % 2) * 280, 330 - (index / 2) * 34, 276, 28);
        box.state = NSControlStateValueOn;
        box.enabled = python != nil;
        box.identifier = keys[index];
        [accessory addSubview:box];
        [boxes addObject:box];
    }
    NSAlert *alert = [NSAlert new];
    alert.messageText = tr(@"gui_uninstall_title");
    alert.informativeText = tr(@"gui_uninstall_explanation");
    alert.accessoryView = accessory;
    NSButton *confirm = [alert addButtonWithTitle:tr(@"gui_uninstall_confirm")];
    confirm.keyEquivalent = @"";
    NSButton *back = [alert addButtonWithTitle:tr(@"gui_cancel")];
    back.keyEquivalent = @"\r";
    [alert beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse response) {
        if (response != NSAlertFirstButtonReturn) {
            [self removeStagedPython];
            [self setBusy:NO];
            self.status.stringValue = tr(@"gui_preserve");
            return;
        }
        NSMutableArray<NSString *> *chosen = [NSMutableArray array];
        for (NSButton *box in boxes) if (box.state == NSControlStateValueOn) [chosen addObject:box.identifier];
        self.status.stringValue = tr(@"gui_uninstall_running");
        if (!python) { [self removeApplication]; return; }
        [self runHelper:@[python, @"-I", @"/Library/Application Support/spaCR/verify_runtime.py", @"uninstall-apply",
                          @"--preview", self.uninstallPreview, @"--categories", [chosen componentsJoinedByString:@","]]
             completion:^(int code, NSString *output) {
            [NSFileManager.defaultManager removeItemAtPath:self.uninstallPreview error:NULL];
            if (code) { [self uninstallStopped]; return; }
            (void)output;
            [self removeApplication];
        }];
    }];
}
- (void)removeApplication {
    self.status.stringValue = tr(@"gui_uninstall_admin");
    NSAppleScript *script = [[NSAppleScript alloc] initWithSource:
        @"do shell script \"case $(/usr/bin/readlink /usr/local/bin/spacr) in /Applications/spaCR.app/*) /bin/rm -f /usr/local/bin/spacr;; esac; "
        @"/bin/rm -rf '/Applications/spaCR.app' '/Library/Application Support/spaCR'; "
        @"/usr/sbin/pkgutil --forget com.einarolafsson.spacr.online >/dev/null 2>&1; exit 0\" with administrator privileges"];
    NSDictionary *error = nil;
    [script executeAndReturnError:&error];
    if (error) {
        [self appendLog:[NSString stringWithFormat:@"%@\n", error[NSAppleScriptErrorMessage] ?: tr(@"gui_uninstall_failed")]];
        [self uninstallStopped];
        return;
    }
    self.removed = YES;
    [self removeStagedPython];
    [self setBusy:NO];
    self.status.stringValue = tr(@"gui_uninstall_done");
}
- (void)uninstallStopped {
    [self removeStagedPython];
    [self setBusy:NO];
    self.status.stringValue = tr(@"gui_uninstall_failed");
}
- (void)cancel:(id)sender {
    (void)sender;
    if (self.child <= 0) { [NSApp terminate:nil]; return; }
    self.cancelling = YES;
    self.status.stringValue = tr(@"gui_stopping");
    self.cancel.enabled = NO;
    if (spacr_install_cancel(self.child) < 0 && errno != ESRCH)
        [self appendLog:[NSString stringWithFormat:@"%s\n", strerror(errno)]];
}
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    (void)sender;
    if (self.child > 0) { [self cancel:nil]; return NSTerminateCancel; }
    return NSTerminateNow;
}
- (BOOL)windowShouldClose:(NSWindow *)sender {
    (void)sender;
    [self cancel:nil];
    return NO;
}
@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc > 1 && !strcmp(argv[1], "--spacr-uninstall-native")) return uninstallNative(argc, argv);
        NSString *locale = NSProcessInfo.processInfo.environment[@"SPACR_INSTALL_LANGUAGE"];
        if (!locale.length) locale = NSLocale.preferredLanguages.firstObject ?: @"en";
        language = [[locale stringByReplacingOccurrencesOfString:@"_" withString:@"-"] componentsSeparatedByString:@"-"].firstObject.lowercaseString;
        if ([language isEqualToString:@"zh"]) language = @"zh_CN";
        NSString *resources = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"installer-i18n"];
        NSData *data = [NSData dataWithContentsOfFile:[resources stringByAppendingPathComponent:[language stringByAppendingString:@".json"]]];
        if (!data) {
            language = @"en";
            data = [NSData dataWithContentsOfFile:[resources stringByAppendingPathComponent:@"en.json"]];
        }
        messages = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
        if (![messages isKindOfClass:NSDictionary.class]) return 2;
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        SpacrInstaller *delegate = [SpacrInstaller new];
        NSMutableArray *forwarded = [NSMutableArray array];
        for (int index = 1; index < argc; ++index) [forwarded addObject:[NSString stringWithUTF8String:argv[index]]];
        delegate.forwardedArguments = forwarded;
        NSApp.delegate = delegate;
        [NSApp run];
    }
    return 0;
}
