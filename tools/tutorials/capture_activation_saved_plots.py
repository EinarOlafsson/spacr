"""Record real Activation outputs in an explicitly external image viewer."""
from capture_barcode_saved_plots import launch


if __name__ == '__main__':
    raise SystemExit(launch('activation', 'activation_saved_plot_viewer',
                           '--activation-saved-plots', 480))
