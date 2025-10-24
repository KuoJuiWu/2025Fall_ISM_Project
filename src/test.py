import numpy as np


# write a function to calculate blackbody radiation intensity
def blackbody_intensity(wavelength, temperature):    
    """
    Calculate the blackbody radiation intensity using Planck's law.

    Parameters:
    wavelength (float or np.ndarray): Wavelength(s) in meters.
    temperature (float): Temperature in Kelvin.

    Returns:
    float or np.ndarray: Intensity in W·sr⁻¹·m⁻³.
    """ 
    h = 6.62607015e-34  # Planck's constant in J·s
    c = 2.99792458e8    # Speed of light in m/s
    k = 1.380649e-23    # Boltzmann's constant in
    