# === Imports ===
import numpy as np  # Array operations
import matplotlib.pyplot as plt  # Plotting
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array # For the color scheme
import pandas as pd  # Data handling
from astropy.coordinates import SkyCoord  # Astronomical coordinate handling
import astropy.units as u  # Physical units 
import scipy.optimize as spopt  # Optimization algorithms
import scipy.stats as st
from scipy.stats import gaussian_kde

# === Color Scheme ===
def discretemap(colormap, hexclrs):
    """
    Produce a colormap from a list of discrete colors without interpolation.
    """
    clrs = to_rgba_array(hexclrs)
    clrs = np.vstack([clrs[0], clrs, clrs[-1]])
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [ (i/(len(clrs)-2.), clrs[i, ki], clrs[i+1, ki]) for i in range(len(clrs)-1) ]
    return LinearSegmentedColormap(colormap, cdict)

class TOLcmaps:
    """
    Simplified TOLcmaps with only 'nightfall_discrete' and 'YlOrBr_discrete'.
    """
    def __init__(self):
        self.cmap = None
        self.cname = None
        self.namelist = ('nightfall_discrete', 'YlOrBr_discrete')
        self.funcdict = {
            'nightfall_discrete': self.__nightfall_discrete,
            'YlOrBr_discrete': self.__YlOrBr_discrete,
            'BuRd_discrete': self.__BuRd_discrete,
        }

    def __nightfall_discrete(self):
        clrs = ['#125A56', '#238F9D', '#60BCE9', '#C6DBED', '#ECEADA',
                '#F9D576', '#FD9A44', '#E94C1F', '#A01813']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __YlOrBr_discrete(self):
        clrs = ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
                '#EC7014', '#CC4C02', '#993404', '#662506']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#888888')
        
    def __BuRd_discrete(self):
        """
        Define colormap 'BuRd_discrete'.
        """
        clrs = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFEE99')

    def show(self):
        print(' '.join(self.namelist))

    def get(self, cname='nightfall_discrete'):
        self.cname = cname
        if cname in self.funcdict:
            self.funcdict[cname]()
            return self.cmap
        else:
            raise ValueError(f"Colormap {cname} is not available. Choose from {self.namelist}.")

def tol_cmap(colormap=None):
    """
    Return matplotlib colormap. Only 'nightfall_discrete', 'YlOrBr_discrete', and 'BuRd_discrete' supported.
    """
    obj = TOLcmaps()
    if colormap is None:
        return obj.namelist
    return obj.get(colormap)

# === Load in MISHAPS File ===
def load_catalog(file_path):
    """Read in our data file and apply column names."""
    if file_path.endswith('.cmd.calibrate_ps'):
        # Load text-based photometric catalog with fixed column order
        column_labels = [
            'ID', 'RA_z', 'Dec_z', 'ID_z', 'X_z', 'Y_z', 'mis_z', 'mis_z_err', 'Sky_z', 'qflag_z', '?1z', '?2z',
            'RA_r', 'Dec_r', 'ID_r', 'X_r', 'Y_r', 'mis_r', 'mis_r_err', 'Sky_r', 'qflag_r', '?1r', '?2r',
            'RA_u', 'Dec_u', 'ID_u', 'X_u', 'Y_u', 'mis_u', 'mis_u_err', 'Sky_u', 'qflag_u', '?1u', '?2u',
            'RA_g', 'Dec_g', 'ID_g', 'X_g', 'Y_g', 'mis_g', 'mis_g_err', 'Sky_g', 'qflag_g', '?1g', '?2g',
            'RA_i', 'Dec_i', 'ID_i', 'X_i', 'Y_i', 'mis_i', 'mis_i_err', 'Sky_i', 'qflag_i', '?1i', '?2i'
        ]
        data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, sep = '\\s+')
        if len(data.columns) >= len(column_labels):
            data.columns = column_labels[:len(data.columns)]
        else:
            raise ValueError(f"Expected at least {len(column_labels)} columns, got {len(data.columns)}")
    else:
        raise ValueError('Unsupported file format. Must be .cmd.calibrate_ps')
    return data

# === Define Parameters and Quality Filters===
def filtered_magnitude(file_path):
    """Return filtered magnitudes and colors from the data file."""
    data = load_catalog(file_path)

    # Filter for valid data and quality flags
    mask_nan = data['mis_r'].notna() & data['mis_z'].notna()
    mask_flag = (data['qflag_r'] == 1) & (data['qflag_z'] == 1)
    mask_err = (data['mis_r_err'] < 0.2) & (data['mis_z_err'] < 0.2)
    combined_mask = mask_nan & mask_flag & mask_err

    data = data[combined_mask]

    if 'mis_r' not in data.columns or 'mis_z' not in data.columns:
        raise ValueError("Missing 'mis_r' or 'mis_z' columns in input file.")

    r_mag = data['mis_r']
    z_mag = data['mis_z']
    rmz_color = r_mag - z_mag

    return r_mag, z_mag, rmz_color, data

# === Red Clump Color Reference Function ===
def red_clump_color(l, b):
    """Estimate intrinsic color and magnitudes of red clump stars for given Galactic coordinates."""
    I_0 = 14.3955 - 0.0239 * l + 0.0122 * np.abs(b)
    I_mag = np.array([I_0, I_0 + 0.737, I_0 - 1.07])  # RC, RGBB, AGBB
    dis_mod = I_mag - 14.3955 + 5.0 * np.log10(8178.0) - 5.0
    distance = 10**(0.2 * (dis_mod + 5.0))
    V_mag = 1.060 + I_mag

    # SDSS transformation
    g_sdss = V_mag + 0.553
    r_sdss = V_mag - 0.327
    i_sdss = V_mag - 0.600
    gmi_sdss = g_sdss - i_sdss

    # SDSS → Pan-STARRS transformation
    gpmgs = -0.01808 - 0.13595 * gmi_sdss + 0.01941 * gmi_sdss**2 - 0.00183 * gmi_sdss**3
    rpmrs = -0.01836 - 0.03577 * gmi_sdss + 0.02612 * gmi_sdss**2 - 0.00558 * gmi_sdss**3
    ipmis = 0.01170 - 0.00400 * gmi_sdss + 0.00066 * gmi_sdss**2 - 0.00058 * gmi_sdss**3

    g_ps1 = g_sdss + gpmgs
    r_ps1 = r_sdss + rpmrs
    i_ps1 = i_sdss + ipmis
    gmi_ps1 = g_ps1 - i_ps1
    rmi_ps1 = r_ps1 - i_ps1
    imz_ps1 = 0.234625 + 0.211715 * (gmi_ps1 - 1.25)
    rmz_ps1 = rmi_ps1 + imz_ps1
    z_ps1 = r_ps1 - rmz_ps1

    return {
        "clump-g": g_ps1[0], "clump-r": r_ps1[0], "clump-i": i_ps1[0], "clump-z": z_ps1[0],
        "rgbb-g": g_ps1[1], "rgbb-r": r_ps1[1], "rgbb-i": i_ps1[1], "rgbb-z": z_ps1[1],
        "agbb-g": g_ps1[2], "agbb-r": r_ps1[2], "agbb-i": i_ps1[2], "agbb-z": z_ps1[2],
        "distance": distance[0], "distance_modulus": dis_mod[0]
    }

# === Find the Position (RA, DEC) ===
def get_mean_position(file_path):
    """Return the mean (RA, Dec) from the data file"""
    data = load_catalog(file_path)
    
    # Try to identify ra and dec columns automatically
    if 'RA_r' in data.columns and 'Dec_r' in data.columns:
        ra_r = data['RA_r']
        dec_r = data['Dec_r']
    
    else:
        raise ValueError("Missing 'RA_r' or 'Dec_r' columns in input file.")

    ra_mean_r = np.nanmean(ra_r)
    dec_mean_r = np.nanmean(dec_r)
    
    return ra_mean, dec_mean

def initialize_red_clump(ra, dec, magnitude_band='r', color_band_2='z'):
    """Compute Galactic coordinates and red clump magnitude references."""
    coords = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    l = coords.galactic.l.value
    b = coords.galactic.b.value
    print(f"Galactic Coordinates: l = {l:.3f}, b = {b:.3f}")

    red_clump = red_clump_color(l, b)

    global rgbbmag0_r, agbbmag0_r, rgbbmag0_z, agbbmag0_z
    
    clumpmag0_r = red_clump["clump-" + magnitude_band]
    clumpmag0_z = red_clump["clump-" + color_band_2]

    rgbbmag0_r = red_clump["rgbb-" + magnitude_band]
    rgbbmag0_z = red_clump["rgbb-" + color_band_2]
    agbbmag0_r = red_clump["agbb-" + magnitude_band]
    agbbmag0_z = red_clump["agbb-" + color_band_2]

    return l, b, red_clump

def auto_color_mag_cut(data, clump_r, clump_z, color_width=0.5, mag_width=1.0):
    """
    Automatically select stars near the expected red clump region in color–magnitude space."""
    
    # Ensure required columns exist
    if 'mis_r' not in data.columns or 'mis_z' not in data.columns:
        raise ValueError("Missing required columns 'mis_r' or 'mis_z' in input data.")
    
    r_mag = data['mis_r']
    z_mag = data['mis_z']
    
    # Compute color and define cut boundaries
    rmz_color = r_mag - z_mag
    color_center = clump_r - clump_z
    mag_center = clump_r
    
    color_cut = (rmz_color > color_center - color_width) & (rmz_color < color_center + color_width)
    mag_cut = (r_mag > mag_center - mag_width) & (r_mag < mag_center + mag_width)
    
    selected = data[color_cut & mag_cut]
    print(f"Selected {len(selected)} stars near predicted red clump region.")
    
    # Optional diagnostic plot
    if show_plot:
        plt.figure(figsize=(6, 5))
        plt.scatter(color, rmag, s=5, c='gray', alpha=0.4, label='All stars')
        plt.scatter(color[color_cut & mag_cut], rmag[color_cut & mag_cut],
                    s=8, c='red', alpha=0.7, label='Selected RC region')

        # Plot rectangle showing cut boundaries
        rect_x = [color_center - color_width, color_center + color_width,
                  color_center + color_width, color_center - color_width, color_center - color_width]
        rect_y = [mag_center - mag_width, mag_center - mag_width,
                  mag_center + mag_width, mag_center + mag_width, mag_center - mag_width]
        plt.plot(rect_x, rect_y, 'r--', lw=1)

        plt.gca().invert_yaxis()  # Magnitude axis increases upward
        plt.xlabel("(r - z) color (mag)")
        plt.ylabel("r magnitude")
        plt.title("Automatic Red Clump Selection")
        plt.legend(loc='best', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    return selected

# === Red Clump Model (Quadratic + Gaussian) for the r-band ===
def rcmodel_r(x, amplitude, slope, magnitude_rc, number_rc, sigma_rc):
    x = np.float64(x)
    A = amplitude*np.exp(slope*(x - magnitude_rc))
    B = number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-magnitude_rc)**2/sigma_rc**2)
    C = 0.201*number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-(magnitude_rc+rgbbmag0_r))**2/sigma_rc**2)
    D = 0.028*number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-(magnitude_rc+agbbmag0_r))**2/sigma_rc**2)
    
    return A + B + C + D    

# === Red Clump Model (Quadratic + Gaussian) for the z-band ===
def rcmodel_z(x, amplitude, slope, magnitude_rc, number_rc, sigma_rc):
    x = np.float64(x)
    A = amplitude*np.exp(slope*(x - magnitude_rc))
    B = number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-magnitude_rc)**2/sigma_rc**2)
    C = 0.201*number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-(magnitude_rc+rgbbmag0_z))**2/sigma_rc**2)
    D = 0.028*number_rc/(np.sqrt(2*np.pi*sigma_rc**2))*np.exp(-0.5*(x-(magnitude_rc+agbbmag0_z))**2/sigma_rc**2)
    return A + B + C + D  

# === Red Clump Gaussian for All Bands ===
def cgauss(x, amplitude, mean, sigma):
    x = np.float64(x)
    
    return amplitude/(np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(x-mean)**2/sigma**2)
