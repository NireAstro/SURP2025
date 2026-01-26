#!/usr/bin/env python3
"""
rcfit_script.py
A reusable script to process photometric catalogs (r,z), select red-clump candidates,
fit the red clump luminosity function, and produce basic plots and outputs.

Usage:
    python rcfit_script.py --input data/catalog.fits --output results/ --band r --magcol MAG_R

This script is a generalized, cleaned-up version suitable for running on multiple files.
It intentionally keeps function boundaries clear so you can import functions into other scripts.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# -------------------- I/O --------------------
def read_catalog(path: str) -> pd.DataFrame:
    """
    Read a photometric catalog. Supports FITS tables and CSV files.
    Returns a pandas DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in ('.fits', '.fit', '.fz'):
        try:
            t = Table.read(str(path))
            df = t.to_pandas()
        except Exception as e:
            # fallback to astropy.io.fits
            with fits.open(str(path)) as hdul:
                data = hdul[1].data
                df = pd.DataFrame(data)
    elif path.suffix.lower() in ('.csv', '.txt'):
        df = pd.read_csv(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df

# -------------------- Preprocessing --------------------
def basic_filter(df: pd.DataFrame,
                 mag_col: str,
                 mag_min: float = -np.inf,
                 mag_max: float = np.inf,
                 err_col: str | None = None,
                 err_max: float = None) -> pd.DataFrame:
    """
    Apply simple magnitude and error filters.
    """
    sel = np.ones(len(df), dtype=bool)
    if mag_col in df.columns:
        sel &= (df[mag_col] > mag_min) & (df[mag_col] < mag_max)
    if err_col and err_col in df.columns and err_max is not None:
        sel &= (df[err_col] <= err_max)
    return df.loc[sel].copy()

# -------------------- Selection --------------------
def construct_color(df: pd.DataFrame, mag1: str, mag2: str, color_name: str = 'color') -> pd.DataFrame:
    """
    Construct a color column = mag1 - mag2.
    """
    df = df.copy()
    df[color_name] = df[mag1] - df[mag2]
    return df

def select_rc_candidates(df: pd.DataFrame,
                         color_col: str,
                         mag_col: str,
                         color_center: float,
                         color_width: float,
                         mag_range: tuple[float, float]) -> pd.DataFrame:
    """
    Select red-clump candidates using a rectangular box in color-magnitude space.
    - color_center, color_width: defines color_center +/- color_width/2
    - mag_range: (mag_min, mag_max)
    """
    cmin = color_center - color_width/2
    cmax = color_center + color_width/2
    mmin, mmax = mag_range
    sel = (df[color_col] >= cmin) & (df[color_col] <= cmax) & (df[mag_col] >= mmin) & (df[mag_col] <= mmax)
    return df.loc[sel].copy()

# -------------------- Fitting --------------------
def gaussian(x, amp, mu, sigma, const):
    return amp * np.exp(-0.5 * ((x - mu)/sigma)**2) + const

def fit_rc_magnitude(df: pd.DataFrame,
                     mag_col: str,
                     nbins: int = 60,
                     fit_range: tuple[float, float] | None = None):
    """
    Fit a Gaussian + constant background to the magnitude histogram (red clump peak).
    Returns fit parameters and histogram data.
    """
    mags = df[mag_col].dropna().values
    if fit_range is None:
        fit_range = (np.nanpercentile(mags, 1), np.nanpercentile(mags, 99))
    hist, edges = np.histogram(mags, bins=nbins, range=fit_range)
    centers = 0.5*(edges[1:] + edges[:-1])

    # initial guesses
    amp0 = hist.max() - np.median(hist)
    mu0 = centers[np.argmax(hist)]
    sigma0 = 0.2
    const0 = np.median(hist)

    try:
        popt, pcov = curve_fit(gaussian, centers, hist, p0=[amp0, mu0, sigma0, const0],
                               bounds=([0, fit_range[0], 0, 0], [np.inf, fit_range[1], np.inf, np.inf]))
    except Exception as e:
        # fallback: fit only a gaussian to the data points without bounds
        popt, pcov = curve_fit(gaussian, centers, hist, p0=[amp0, mu0, sigma0, const0])
    perr = np.sqrt(np.diag(pcov))
    return dict(popt=popt, perr=perr, hist=hist, centers=centers, edges=edges)

# -------------------- Plotting --------------------
def plot_cmd(df: pd.DataFrame, color_col: str, mag_col: str, outpath: str | None = None, title: str = ""):
    plt.figure(figsize=(6,6))
    plt.scatter(df[color_col], df[mag_col], s=6, alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel(color_col)
    plt.ylabel(mag_col)
    plt.title(title)
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_hist_with_fit(centers, hist, popt, outpath: str | None = None, title: str = ""):
    plt.figure(figsize=(6,4))
    plt.bar(centers, hist, width=(centers[1]-centers[0]), align='center', alpha=0.6, label='hist')
    x = np.linspace(centers.min(), centers.max(), 400)
    plt.plot(x, gaussian(x, *popt), lw=2, label='Gaussian+const')
    plt.xlabel('Magnitude')
    plt.ylabel('Counts')
    plt.gca().invert_xaxis()  # optional depending on convention
    plt.title(title)
    plt.legend()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# -------------------- Utilities --------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# -------------------- Main pipeline --------------------
def run_pipeline(input_file: str,
                 output_dir: str,
                 mag_r: str = 'MAG_R',
                 mag_z: str = 'MAG_Z',
                 color_center: float = 0.6,
                 color_width: float = 0.4,
                 mag_range: tuple[float, float] = (11.0, 16.0),
                 nbins: int = 60):
    ensure_dir(output_dir)
    df = read_catalog(input_file)

    # If specified columns don't exist, try common alternatives
    for col in (mag_r, mag_z):
        if col not in df.columns:
            # try lowercase, or typical names
            alt = [col.lower(), col.replace('MAG_','mag_'), col.replace('MAG_','')]
            for a in alt:
                if a in df.columns:
                    df.rename(columns={a: col}, inplace=True)
                    break

    # Build color column
    if mag_r in df.columns and mag_z in df.columns:
        df = construct_color(df, mag_r, mag_z, color_name='r_minus_z')
        color_col = 'r_minus_z'
        mag_col = mag_z  # often RC is fit in z-band; choose as default
    else:
        # fallback: if only one band present, try single-band processing
        raise ValueError("Both magnitude columns required (r and z) for color construction. Found: " + ", ".join(df.columns))

    # Basic filtering (example)
    df_filt = basic_filter(df, mag_col, mag_min=mag_range[0]-2, mag_max=mag_range[1]+2)

    # Select RC candidates
    rc = select_rc_candidates(df_filt, color_col=color_col, mag_col=mag_col,
                              color_center=color_center, color_width=color_width,
                              mag_range=mag_range)

    # Fit RC magnitude distribution
    fit = fit_rc_magnitude(rc, mag_col=mag_col, nbins=nbins, fit_range=mag_range)

    # Save fit results
    result = {
        'input_file': input_file,
        'n_total': len(df),
        'n_filtered': len(df_filt),
        'n_rc_candidates': len(rc),
        'fit_mu': float(fit['popt'][1]),
        'fit_sigma': float(fit['popt'][2]),
        'fit_amp': float(fit['popt'][0]),
        'fit_const': float(fit['popt'][3])
    }
    out_table = pd.DataFrame([result])
    out_table.to_csv(os.path.join(output_dir, 'rc_fit_summary.csv'), index=False)

    # Make plots
    plot_cmd(df_filt, color_col, mag_col, outpath=os.path.join(output_dir, 'cmd_filtered.png'),
             title=f'CMD (filtered) {Path(input_file).name}')
    plot_cmd(rc, color_col, mag_col, outpath=os.path.join(output_dir, 'cmd_rc_candidates.png'),
             title=f'RC candidates {Path(input_file).name}')
    plot_hist_with_fit(fit['centers'], fit['hist'], fit['popt'],
                       outpath=os.path.join(output_dir, 'rc_hist_fit.png'),
                       title=f'RC magnitude fit {Path(input_file).name}')

    print("Pipeline finished. Summary:")
    print(out_table.to_string(index=False))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run red-clump fitting pipeline on a photometric catalog.")
    p.add_argument('--input', '-i', required=True, help='Input catalog (FITS or CSV).')
    p.add_argument('--output', '-o', required=True, help='Output directory.')
    p.add_argument('--mag-r', default='MAG_R', help='Column name for r-band magnitude.')
    p.add_argument('--mag-z', default='MAG_Z', help='Column name for z-band magnitude.')
    p.add_argument('--color-center', type=float, default=0.6, help='Center of color selection (r-z).')
    p.add_argument('--color-width', type=float, default=0.4, help='Width of color selection (r-z).')
    p.add_argument('--mag-min', type=float, default=11.0, help='Minimum magnitude for RC selection (z).')
    p.add_argument('--mag-max', type=float, default=16.0, help='Maximum magnitude for RC selection (z).')
    p.add_argument('--nbins', type=int, default=60, help='Number of bins for histogram fit.')
    return p.parse_args()

def main():
    args = parse_args()
    run_pipeline(args.input, args.output,
                 mag_r=args.mag_r, mag_z=args.mag_z,
                 color_center=args.color_center, color_width=args.color_width,
                 mag_range=(args.mag_min, args.mag_max),
                 nbins=args.nbins)

if __name__ == '__main__':
    main()
