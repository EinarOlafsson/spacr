def gui_mask():
    from .cli import get_arg_parser
    from .version import version_str

    args = get_arg_parser().parse_args()
    
    if args.version:
        print(version_str)
        return

    if args.headless:
        settings = {}
        spacr.core.preprocess_generate_masks(settings['src'], settings=settings, advanced_settings={})
        return

    global vars_dict, root
    root, vars_dict = initiate_mask_root(1000, 1500)
    root.mainloop()