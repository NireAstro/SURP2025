import pandas as pd
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_catalog(file_path):
    """Load catalog from FITS, CSV, or space-delimited text file."""
    if file_path.endswith('.fits'):
        data = Table.read(file_path).to_pandas()
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.cmd.calibrate_ps'):
        # Load text-based photometric catalog with fixed column order
        column_labels = [
            'ID', 'RA_z', 'Dec_z', 'ID_z', 'X_z', 'Y_z', 'mis_z', 'mis_z_err', 'Sky_z', 'qflag_z', '?1z', '?2z',
            'RA_r', 'Dec_r', 'ID_r', 'X_r', 'Y_r', 'mis_r', 'mis_r_err', 'Sky_r', 'qflag_r', '?1r', '?2r',
            'RA_u', 'Dec_u', 'ID_u', 'X_u', 'Y_u', 'mis_u', 'mis_u_err', 'Sky_u', 'qflag_u', '?1u', '?2u',
            'RA_g', 'Dec_g', 'ID_g', 'X_g', 'Y_g', 'mis_g', 'mis_g_err', 'Sky_g', 'qflag_g', '?1g', '?2g',
            'RA_i', 'Dec_i', 'ID_i', 'X_i', 'Y_i', 'mis_i', 'mis_i_err', 'Sky_i', 'qflag_i', '?1i', '?2i'
        ]
        data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)
        if len(data.columns) >= len(column_labels):
            data.columns = column_labels[:len(data.columns)]
        else:
            raise ValueError(f"Expected at least {len(column_labels)} columns, got {len(data.columns)}")
    else:
        raise ValueError('Unsupported file format. Must be .fits, .csv, or .cmd.calibrate_ps')
    return data


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def run_rcfit(file_path):
    """Run red clump fitting procedure on an input photometric catalog."""
    data = load_catalog(file_path)

    # Try to identify magnitude columns automatically
    if 'mis_r' in data.columns and 'mis_z' in data.columns:
        rmag = data['mis_r']
        zmag = data['mis_z']
    
    else:
        raise ValueError("Missing 'mis_r' or 'mis_z' columns in input file.")

    color = rmag - zmag

    # Filter out NaNs and extreme values
    mask = np.isfinite(rmag) & np.isfinite(zmag) & np.isfinite(color)
    rmag, zmag, color = rmag[mask], zmag[mask], color[mask]

    # Select likely RC candidates (rough cuts)
    rc_mask = (color > 0.4) & (color < 1.2) & (zmag > 15) & (zmag < 20)
    rc_color, rc_mag = color[rc_mask], zmag[rc_mask]

    # Fit Gaussian to z-band magnitudes
    counts, bins = np.histogram(rc_mag, bins=30)
    centers = 0.5 * (bins[1:] + bins[:-1])
    p0 = [counts.max(), rc_mag.mean(), rc_mag.std()]
    try:
        popt, pcov = curve_fit(gaussian, centers, counts, p0=p0)
        amp, mu, sigma = popt
        fit_success = True
    except Exception:
        amp = mu = sigma = np.nan
        fit_success = False

    # Plot CMD and histogram
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(color, zmag, s=5, alpha=0.4)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('r - z')
    ax[0].set_ylabel('z mag')
    ax[0].set_title('Color-Magnitude Diagram')

    ax[1].bar(centers, counts, width=(bins[1]-bins[0]), color='gray', alpha=0.6, label='Data')
    if fit_success:
        ax[1].plot(centers, gaussian(centers, *popt), 'r-', label='Gaussian Fit')
    ax[1].set_xlabel('z mag')
    ax[1].set_ylabel('Counts')
    ax[1].legend()
    ax[1].set_title('RC Magnitude Distribution')

    plt.tight_layout()
    plt.savefig('rcfit_results.png', dpi=150)
    plt.close()

    return {
        'mean_mag': mu,
        'sigma_mag': sigma,
        'num_rc_stars': len(rc_mag),
        'fit_success': fit_success
    }
