"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu and Michael Rariden.
"""

import argparse

import argparse


def get_arg_parser():
    """ Parses command line arguments for spacr main functions

    Note: this function has to be in a separate file to allow autodoc to work for CLI.
    The autodoc_mock_imports in conf.py does not work for sphinx-argparse sometimes,
    see https://github.com/ashb/sphinx-argparse/issues/9#issue-1097057823
    """
    
    parser = argparse.ArgumentParser(description="SPACR Mask App Command Line Parameters")
    hardware_args = parser.add_argument_group("Hardware Arguments")
    input_img_args = parser.add_argument_group("Input Image Arguments")
    #model_args = parser.add_argument_group("Model Arguments")
    #algorithm_args = parser.add_argument_group("Algorithm Arguments")
    #training_args = parser.add_argument_group("Training Arguments")
    #output_args = parser.add_argument_group("Output Arguments")
    
    # misc settings
    parser.add_argument("--version", action="store_true",
                        help="show version info")
    # misc settings
    parser.add_argument("--headless", action="store_true",
                        help="run the app without the gui")
    
    parser.add_argument("--verbose", action="store_true",
                        help="show information about running and settings and save to log")
    
    hardware_args.add_argument("--gpu_device", required=False, default="0", type=str,
                               help="which gpu device to use, use an integer for torch, or mps for M1")
    
    input_img_args.add_argument("--src", default=[], type=str,
                                help="folder containing data to run or train on.")
    return parser