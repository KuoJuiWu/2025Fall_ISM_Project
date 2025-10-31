import numpy as np
import glob
import os 
from ExtractSpec import *

# Change to the directory containing the FITS files
os.chdir('/media/kuo-jui/data/ISM_finalproject/')
fitls = glob.glob('*.fits')
#print(fitls)
data_dict = create_data_dict(fitls,check_header= False, check_file_header='TMC1A_g140h-f100lp_s3d.fits')

# For example, plot the raw image, g140 
g140_s3d = data_dict['TMC1A_g140h.s3d']['data']
header = data_dict['TMC1A_g140h.s3d']['header']
plot_image(g140_s3d[2855], header, vmin = -1, vmax = 1e3, a = 1e-2, text = '-70km/s', savefig = False, save_name = 'g140_2865.png')
wave, flux = calc_spec(g140_s3d, header = header)

# Plot the FeII 1.644 micron line spectrum
data = g140_s3d
plot_line_spectrum(wave, header, flux, xlim = (-800,800), ylim = (-2e-3, 2e-2), text = ('1.644', 'FeII'), restwave = 1.644, savefig = False, save_name = 'FeII1644_spectrum.png', plot_all_spec= True)
cont_spaxel, line_spaxel = generate_line_cont(data, wave)

# Plot the continuum emission of the FeII 1.644
plot_image(line_spaxel[2866], header, vmin = -1, vmax = 1e3, a = 1e-2, text = '-70km/s', savefig = False, save_name = 'line_2866.png')
plot_image(cont_spaxel[2866], header, vmin = -1, vmax = 1e3, a = 1e-2, text = '-70km/s', savefig = False, save_name = 'cont_2866.png')
cont, flux_line = calc_contsubtract_spec(line_spaxel, cont_spaxel, header = header)

# Plot the comparison of line spectrum and continuum spectrum
v_1644 = optical_veocity(wave, header= header, restwave = 1.644)
plot_comparison_line_continuum(v_1644, wave, flux, flux_line, cont, plot_all_spec = True, savefig = False, save_name = 'Comparison_line_cont.png')