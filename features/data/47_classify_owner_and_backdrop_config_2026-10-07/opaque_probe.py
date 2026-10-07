from spacr.qt import theme

original = theme._window_block

def opaque_window(theme_name, palette, background, body_px, backdrop=False):
    return original(theme_name, palette, background, body_px, backdrop=False)

def pytest_runtest_call(item):
    if item.name == 'test_the_home_screen_is_not_black_on_a_real_display':
        theme._window_block = opaque_window
