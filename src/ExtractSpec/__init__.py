# src/ExtractSpec/__init__.py
from .utils import read_fits, create_data_dict, optical_veocity, plot_image, plot_line_spectrum, calc_cont, generate_line_cont, calc_contsubtract_spec, plot_comparison_line_continuum, calc_spec

__all__ = ["read_fits", "create_data_dict", "optical_veocity", "plot_image", "plot_line_spectrum", "calc_cont", "generate_line_cont", "calc_contsubtract_spec", "plot_comparison_line_continuum", "calc_spec"]
__version__ = "0.1.0"