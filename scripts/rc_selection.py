# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

# === Load in MISHAPS File ===

def load_catalog(file_path):
    """Read in our data file and apply column names."""
    if not file_path.endswith('.cmd.calibrate_ps'):
        raise ValueError("Unsupported file format. Must be .cmd.calibrate_ps")

    column_labels = ['ID', 'RA_z', 'Dec_z', 'ID_z', 'X_z', 'Y_z', 'mis_z', 'mis_z_err', 'Sky_z', 'qflag_z', '?1z', '?2z',
       'RA_r', 'Dec_r', 'ID_r', 'X_r', 'Y_r', 'mis_r', 'mis_r_err', 'Sky_r', 'qflag_r', '?1r', '?2r',
       'RA_u', 'Dec_u', 'ID_u', 'X_u', 'Y_u', 'mis_u', 'mis_u_err', 'Sky_u', 'qflag_u', '?1u', '?2u',
       'RA_g', 'Dec_g', 'ID_g', 'X_g', 'Y_g', 'mis_g', 'mis_g_err', 'Sky_g', 'qflag_g', '?1g', '?2g',
       'RA_i', 'Dec_i', 'ID_i', 'X_i', 'Y_i', 'mis_i', 'mis_i_err', 'Sky_i', 'qflag_i', '?1i', '?2i',
       'RA_Y', 'Dec_Y', 'ID_Y', 'X_Y', 'Y_Y', 'mis_Y', 'mis_Y_err', 'Sky_Y', 'qflag_Y', '?1Y', '?2Y']

    data = pd.read_csv(
        file_path,
        comment='#',
        header=None,
        sep=r'\s+'
    )

    if len(data.columns) < len(column_labels):
        raise ValueError(f"Expected ≥ {len(column_labels)} columns, got {len(data.columns)}")

    data.columns = column_labels[:len(data.columns)]
    return data

# === Photometric Filtering ===

def filtered_magnitude(file_path, err_cut=0.2):
    """Return filtered magnitudes, colors, and filtered DataFrame."""
    data = load_catalog(file_path)

    mask = (
        data['mis_r'].notna() &
        data['mis_z'].notna() &
        (data['qflag_r'] == 1) &
        (data['qflag_z'] == 1) &
        (data['mis_r_err'] < err_cut) &
        (data['mis_z_err'] < err_cut)
    )

    data = data.loc[mask].copy()

    data['rmz'] = data['mis_r'] - data['mis_z']
    return data

# === Red Clump Reference Model

def red_clump_color(l, b):
    """
    Predict observed-frame PS1 magnitudes for the red clump
    using empirical relations.
    """

    I0 = 14.3955 - 0.0239 * l + 0.0122 * np.abs(b)
    I_mag = np.array([I0, I0 + 0.737, I0 - 1.07])  # RC, RGBB, AGBB

    # Distance modulus and distance (pc)
    mu = I_mag - 14.3955 + 5.0 * np.log10(8178.0) - 5.0
    distance = 10 ** (0.2 * (mu + 5.0))

    # Approximate transformations
    V_mag = 1.060 + I_mag

    g_sdss = V_mag + 0.553
    r_sdss = V_mag - 0.327
    i_sdss = V_mag - 0.600

    gmi_sdss = g_sdss - i_sdss

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
        "clump-r": r_ps1[0],
        "clump-z": z_ps1[0],
        "rgbb-r": r_ps1[1],
        "rgbb-z": z_ps1[1],
        "agbb-r": r_ps1[2],
        "agbb-z": z_ps1[2],
        "distance_pc": distance[0],
        "distance_modulus": mu[0]
    }

# === Mean Position & Galactic Coordinates ===

def get_mean_position(data):
    """Return mean RA, Dec (degrees)."""
    ra = np.nanmean(data['RA_r'])
    dec = np.nanmean(data['Dec_r'])
    return ra, dec


def initialize_red_clump(data):
    """Compute Galactic coordinates and RC reference magnitudes."""
    ra, dec = get_mean_position(data)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    l = coords.galactic.l.value
    b = coords.galactic.b.value

    rc_model = red_clump_color(l, b)

    print(f"Mean position: RA={ra:.3f}, Dec={dec:.3f}")
    print(f"Galactic coords: l={l:.2f}, b={b:.2f}")

    return rc_model

# === Plot CMD ===

def plot_cmd(data):
    """Create a color-magnitude diagram."""
    
    rmag = data['mis_r']
    color = data['rmz']

   
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(color, rmag, color = 'black', s=1, alpha=0.2, label='All stars')

    plt.gca().invert_yaxis()
    plt.xlabel(r'$(r - z)$')
    plt.ylabel(r'$r$')
    plt.legend()
    plt.title("Color-Magnitude Diagram")
    plt.tight_layout()
    plt.show()
    
    return None
