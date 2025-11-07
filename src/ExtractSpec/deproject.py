import numpy as np
from scipy.ndimage import map_coordinates

def deproject_image(img:np.ndarray, inc_deg: float, pa_deg: float, x0: float|None =None, y0: float|None = None, order = 1, cval = np.nan):
    """
    Deproject a 2D astronomy image assuming a thin disk.
    - img: 2D numpy array
    - inc_deg: inclination in degrees (0 = face-on)
    - pa_deg: position angle in degrees (E of N)
    - x0,y0: disk center in pixels (if None, uses image max)
    - order: spline interpolation order (0..5); 1 is bilinear
    - cval: fill value for outside points
    Returns: deproj_img (same shape), r_map (pixels), theta_map (radians)
    """
    ny, nx = img.shape
    if x0 is None or y0 is None:
        y0, x0 = np.unravel_index(np.nanargmax(img), img.shape)

    inc = np.deg2rad(inc_deg)
    pa  = np.deg2rad(pa_deg)

    # pixel grid in *output* (deprojected) frame
    y_idx, x_idx = np.indices(img.shape)
    x = x_idx - x0
    y = y_idx - y0

    # Forward transform we want in the *input* (observed) frame:
    # 1) compress y by cos(i) to go from observed -> deproj;
    #    for resampling we need the inverse: stretch y' by 1/cos(i).
    # 2) rotate by -PA to align disk major axis with +x'.
    cosi = np.cos(inc)
    # Inverse mapping to source (observed) coordinates:
    xp =  np.cos(pa)*x + np.sin(pa)*y
    yp = -np.sin(pa)*x + np.cos(pa)*y
    yp_obs = yp * cosi  # undo the deprojection stretch

    # Rotate back to observed frame:
    x_src =  np.cos(pa)*xp - np.sin(pa)*yp_obs + x0
    y_src =  np.sin(pa)*xp + np.cos(pa)*yp_obs + y0

    # Sample from the original image
    coords = np.vstack([y_src.ravel(), x_src.ravel()])
    deproj = map_coordinates(img, coords, order=order, cval=cval).reshape(img.shape)

    # Also return polar coords in deprojected plane
    #r_map = np.hypot(xp, yp)              # pixels in deprojected plane
    #theta_map = np.arctan2(yp, xp)        # radians, deprojected plane

    return deproj #, r_map, theta_map

