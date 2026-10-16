#import <Cocoa/Cocoa.h>
#include "macos_gui_process.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 760, 610)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        backing:NSBackingStoreBuffered defer:NO];
    self.window.title = tr(@"installer_title");
    self.window.delegate = self;
    [self.window center];
    [self label:tr(@"gui_explanation") frame:NSMakeRect(24, 548, 712, 44)];
    [self label:tr(@"gui_public_reports") frame:NSMakeRect(24, 476, 712, 64)];
    self.choices = [NSMutableArray array];
    NSArray *keys = @[@"nsis_gpu", @"gui_share_diagnostics", @"gui_report_issues"];
    for (NSUInteger index = 0; index < keys.count; ++index) {
        NSButton *choice = [NSButton checkboxWithTitle:tr(keys[index]) target:nil action:nil];
        choice.frame = NSMakeRect(24, 428 - index * 44, 712, 40);
        [(NSCell *)choice.cell setWraps:YES];
        choice.toolTip = choice.title;
        choice.state = index == 0 ? NSControlStateValueOn : NSControlStateValueOff;
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
    self.install.enabled = !busy;
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
    NSArray *flags = @[@"--share-diagnostics", @"--report-issues"];
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
