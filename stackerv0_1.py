#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:28:37 2025

@author: philipp
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from astropy.io import fits
from astropy import units as u
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import string
import random
import math
from specutils import Spectrum, SpectralRegion
from specutils.manipulation import FluxConservingResampler
import pandas as pd
import time
import plotly.express as px

def round_sig_down(x, n):
    return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

def round_sig_up(x, n):
    return round(x, n - int(math.ceil(math.log10(abs(x)))) - 1)

def save_plot(folder_path, save_name, base_name, id, data, file_type="png"):
    
    
    # Zielordner erstellen
    plot_folder = os.path.join(folder_path, save_name)
    os.makedirs(plot_folder, exist_ok=True)
    
    # Dateinamen bauen
    if id == "stacked":
        filename_png = "stacked.png"
        filename_html = "stacked.html"
    else:
        filename_png = f"{base_name}_{id}.png"
        filename_html = f"{base_name}_{id}.html"
    
    png_path = os.path.join(plot_folder, filename_png)
    html_path = os.path.join(plot_folder, filename_html)
    
    # Plot speichern
    try:
        plt.figure(figsize=(20, 7))
        plt.plot(data["wavelength"], data["flux"], linewidth=1)
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
    except:
        raise Exception("err: inserted data has no wavelength and flux columns!")
        
    
    try:
        fig = px.line(data, x="wavelength", y="flux", title=filename_html)
        fig.write_html(html_path)
    except:
        raise Exception("err: inserted data has no wavelength and flux columns!")
        
    
    
    
    return png_path, html_path

def wavelength_values(file_path):
    
    step_lengths = []
    start_values = []
    end_values = []
    
    for data in file_path:
        
        with fits.open(data) as f:
            specdata = f[0].data
            crval1 = f[0].header['CRVAL1']
            cdelt1 = f[0].header['CDELT1']
            entries = len(specdata)
        
        step_lengths.append(cdelt1)
        start_values.append(crval1)
        
        end_wavelength = crval1 + cdelt1 * entries
        end_values.append(end_wavelength)
        
    max_step = max(step_lengths)
    min_length = min(start_values)
    max_length = max(end_values)
    
    print(f"Max steplength from all spectra is: {max_step}")
    
    return max_step, min_length, max_length

def stacking(file_path, specUnit, fluxUnit, binfactor, z=1, plotting=False, save_name=False, min_Res = 10000):
    
    if not save_name:
        save_name = time.strftime("%Y%m%d-%H%M%S")
    
    fluxcon = FluxConservingResampler()
    folder_path = os.getcwd()

    data_df = []
    id_counter = 1

    spec_unit = u.Unit(specUnit)
    flux_unit = u.Unit(fluxUnit)

    for data in file_path:
        
        fits_name = os.path.basename(data).replace(".fits", "")
        
        with fits.open(data) as f:
            specdata = f[0].data
            crval1 = f[0].header['CRVAL1']
            cdelt1 = f[0].header['CDELT1']
            # spaltweite
            entries = len(specdata)

        # Calculate resolution 
        if min_Res < 10000:
            print(f"The resolution of spectrum {os.path} is too small! It will not be used in the stack!")
            pass
        
        # Create cdelt table to identify the biggest step
        
        # Wavelength array
        wavelength = (crval1 + np.arange(entries) * cdelt1) / (1 + z)
        wavelength *= spec_unit

        # Flux (byteswap fix)
        flux = np.array(specdata)
        flux = flux.byteswap().view(flux.dtype.newbyteorder("="))
        flux *= flux_unit

        # Remove zeros and NaN
        mask = (~np.isnan(flux)) & (flux != 0)
        wavelength = wavelength[mask]
        flux = flux[mask]

        if plotting:
            
            temp_data = pd.DataFrame({
                        "wavelength": wavelength.value,
                        "flux": flux.value
                        })
            
            png_path, html_path = save_plot(folder_path, save_name, fits_name, id_counter, temp_data)
            
            print(f"PNG Figure saved under path: {png_path}")
            print(f"HTML Figure saved under path: {html_path}")
            
            id_counter += 1


        # Create Spectrum1D
        input_spec = Spectrum(
            spectral_axis=wavelength,
            flux=flux
        )

        # New dispersion grid
        
        step_length, min_length, max_length = wavelength_values(file_path)
        
        # Step width: round to 1 significant figure
        step_width = round_sig_down(step_length, 1) * binfactor
        
        wave_start = round_sig_down(min_length, 2)
        wave_stop = round_sig_up(max_length, 2)

        new_disp_grid = np.arange(wave_start,
                                  wave_stop,
                                  step_width) * spec_unit

        # Resample
        new_spec = fluxcon(input_spec, new_disp_grid)

        # Store resampled data as floats
        df = pd.DataFrame({
            "wavelength": new_spec.spectral_axis.value,
            "flux": new_spec.flux.value
        })

        data_df.append(df)

    # Combine all spectra
    combined = pd.concat(data_df, ignore_index=True)

    # Sort by wavelength
    combined.sort_values("wavelength", inplace=True)

    # Stack (mean per wavelength)
    stacked = combined.groupby("wavelength").mean().reset_index()

    stacked_save = save_plot(folder_path, save_name, "stacked", "stacked", stacked)
    
    print(f"Stacked image saved under path: {stacked_save}")
    

    # Convert back to quantities
    stacked["wavelength"] = stacked["wavelength"] * spec_unit
    stacked["flux"] = stacked["flux"] * flux_unit
    
    

    return stacked, combined, temp_data


"""file_path = ["/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T06_48_37.662/5253-SLb-D2_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T07_06_04.105/5253-SLb-D2_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T07_23_34.538/5253-SLb-D2_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T07_43_30.716/5253-SLb-D1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T07_50_18.513/5253-SLb-D1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_1/UVES.2003-03-31T07_57_04.537/5253-SLb-D1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2003-07-27T00_20_21.049/NGC_5253_DIC2_FLUXCAL_SCI_POINT_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2003-09-04T23_49_26.189/NGC_5253_DIC2_FLUXCAL_SCI_POINT_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2003-09-05T00_15_48.778/NGC_5253_DIC2_FLUXCAL_SCI_POINT_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2003-09-05T00_41_27.487/NGC_5253_DIC2_FLUXCAL_SCI_POINT_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-24T23_40_07.683/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_04_38.875/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_31_56.806/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_39_49.599/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_04_19.647/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_BLUE.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_30_27.940/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_BLUE.fits"
             ] """ # blue Spectra
    
file_path = ["/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2003-09-05T00_41_17.402/NGC_5253_DIC2_FLUXCAL_SCI_POINT_REDU.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-24T23_40_06.724/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDL.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-24T23_40_06.724/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDU.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_04_49.552/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDL.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_04_49.552/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDU.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_31_55.792/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDL.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_31_55.792/NGC_5253_DIC1_FLUXCAL_SCI_EXTND_REDU.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_39_47.342/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDL.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T00_39_47.342/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDU.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_04_28.339/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDL.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_04_28.339/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDU.fits",
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_30_26.983/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDL.fits", 
             "/Users/philipp/Documents/ESOTest/Output/NGC5253_spectra_2/UVES.2004-04-25T01_30_26.983/NGC_5253_DIC2_FLUXCAL_SCI_EXTND_REDU.fits",
    ] #red Specta (partially)


stacked_data, combined, TD = stacking(file_path, "AA", "erg cm-2 s-1 AA-1", 5, 0.001358, plotting=True)

plt.figure(figsize=(20, 7))
plt.plot(stacked_data['wavelength'], stacked_data['flux'], marker = '', linestyle = '-', linewidth=1)
