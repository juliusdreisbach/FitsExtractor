"""
© 2026 Julius Richard Dreisbach – FITS Extractor Utility
Designed for rapid spectrum extraction and normalization.

You may use, copy, and modify this software for personal or educational purposes.
Commercial use is not allowed without permission from the author.
No warranty is provided.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD, DND_TEXT
from astropy.io import fits
from astropy import units as u
import numpy as np
import os
import matplotlib.pyplot as plt
import string
import random
import math
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go


version = "Stacker 1.0"

obj_key = 'OBJECT'
wv_unit_key = 'TUNIT1'
wave_key = 'WAVE'
flux_key = 'FLUX'
err_key = 'ERR_FLUX'
continuum_key = 'CONTINUUM'
status_key = 'STATUS'

show_maximum_files = 25

# Set a folder to save the files into.
folder_name = "extracted"

class FileSizeApp(tk.Toplevel):

    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title(f"FITS Extractor {version}")
        self.geometry("350x250")
        self.dnd_text = "Drag & Drop the .fits file or directory here"
        self.all_loaded_files = []
        self.drop_disabled = False
        self.key_values = [wave_key, flux_key, err_key, continuum_key, status_key]
        self._build_ui()

    def _build_ui(self):

        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side="left", padx=10, pady=10)

        ttk.Label(self.left_frame, text="How To Use: (1) Provide a file directory (2) Stack spectra", font=("Arial", 9)).pack(pady=5)

        self.drop_frame = ttk.Frame(self.left_frame, width=400, height=100, relief="solid", borderwidth=1)
        self.drop_frame.pack(pady=5)
        self.drop_frame.pack_propagate(False)

        self.drop_label = ttk.Label(self.drop_frame, text=self.dnd_text, font=("Arial", 11))
        self.drop_label.pack(expand=True)

        # Drop-Events aktivieren
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.on_drop)

        self.path_label = ttk.Label(self.left_frame, text="Path: -", font=("Arial", 8))
        self.path_label.pack(pady=2)

        # GUI frame for file information
        self.file_frame = ttk.Frame(self.left_frame)
        self.file_frame.pack(pady=5)

        # label for file amount
        self.file_label = ttk.Label(self.file_frame, text="Files loaded: none", font=("Arial", 11))
        self.file_label.pack(side="left", padx=(0, 10))

        # label for file size
        self.result_label = ttk.Label(self.file_frame, text="", font=("Arial", 11))
        self.result_label.pack(side="left", padx=(0, 10))

        # show all files button
        self.show_all_files_button = ttk.Button(self.file_frame, text="Show Files", command=self.show_all_files)
        self.show_all_files_button.pack(side="right")
        
        # stacking button
        self.stacking_frame = ttk.Frame(self.left_frame)
        self.stacking_frame.pack(pady=5)
        
        self.do_individual_plot = tk.BooleanVar(value=False)
        self.do_individual_plot_checkbox = ttk.Checkbutton(
            self.stacking_frame,
            text="Save individual spectra",
            variable=self.do_individual_plot
            )
        self.do_individual_plot_checkbox.pack(side="left", padx=5)
        
        self.stacking = ttk.Button(self.stacking_frame, text="Stack spectra", command=self.do_stacking)
        self.stacking.pack(side="right", padx=(0, 5))
        
        # No functionality?
        """ 
        self.isESO = tk.BooleanVar(value=False)
        self.isESO_checkbox = ttk.Checkbutton(
            self.stacking_frame,
            text="ESO files",
            variable=self.isESO
            )
        self.isESO_checkbox.pack(side="left")
        """
        """
        self.stacking_progress_bar = ttk.Progressbar(self.stacking_frame)
        self.stacking_progress_bar.pack(side="right")
        """ # can not be done because iteration is needed and the function for stacking itself itereates -> can not update progressbar

    def get_size(self, path, no=0):
        """Berechnet die Größe einer Datei oder eines Ordners (in Bytes)."""
        if os.path.isfile(path):
            return os.path.getsize(path), no+1, [path]
        names = [] # all file paths
        total = 0 # total size
        total_no = no # total number of files
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    names.append(fp)
                    total += os.path.getsize(fp)
                    total_no += 1
        return total, total_no, names

    def on_drop(self, event):
        if self.drop_disabled:
            return
        # Called when something is dropped into drag & drop area
        raw_path = event.data.strip()
        path = raw_path.strip("{}")

        s_maxlength = 60
        s = path[:s_maxlength] + "..." if len(path) > s_maxlength else path
        self.path_label.config(text=f"Path: {s}")

        if not os.path.exists(path):
            self.result_label.config(text="err: Invalid path")
            return

        size_bytes, no_of_files, names = self.get_size(path)
        self.all_loaded_files = names
        threshold = 1024
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1000
        size_gb = size_mb / 1000
        if size_bytes < threshold:
            self.result_label.config(text=f"Size: {size_bytes:.2f} bytes")
        elif size_kb < threshold:
            self.result_label.config(text=f"Size: {size_kb:.2f} KB")
        elif size_mb < threshold:
            self.result_label.config(text=f"Size: {size_mb:.2f} MB")
        else:
            self.result_label.config(text=f"Size: {size_gb:.2f} GB")

        self.file_label.config(text=f"Files loaded: {no_of_files}")

    def show_all_files(self):
        msg = ""
        files = len(self.all_loaded_files)
        if self.all_loaded_files == []:
            msg = "No files are loaded yet."
        else:
            for i in range(files):
                msg += f"{os.path.basename(self.all_loaded_files[i])}\n"
                if i > show_maximum_files:
                    msg += f"... ({files - i} more)"
                    break

        tk.messagebox.showinfo(title="List of Loaded Files", message=msg)
    
    def do_stacking(self):
        if not self.all_loaded_files:
            messagebox.showerror("Error", "No files loaded.")
            return
        
        if len(self.all_loaded_files) < 2:
            messagebox.showerror("Error", "Stacking a single spectrum is not useful.")
            return
        
        try:
            save_path = stacking(
                self.all_loaded_files,
                specUnit="AA",
                fluxUnit="erg cm-2 s-1 AA-1",
                binfactor=1,
                z=0,
                single_plotting=self.do_individual_plot.get()
            )
            messagebox.showinfo("Done", f"Data is saved at location: {save_path}.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self.on_closing()

    def on_closing(self):
        self.destroy()

def check_file(file_path):
    print(f"Checking for compatibility: {file_path}", end = ' ')
    filename_full = os.path.basename(file_path)
    filename_splits = filename_full.split(".")
    try:
        filename = filename_splits[len(filename_splits)-2]
        ending = filename_splits[len(filename_splits)-1]
    except:
        filename = filename_full
        ending = "not_given"

    if not ending in ['fits','FITS']:
        print("warn: File is not compatible!")
        return False

    try:
        hdul = fits.open(file_path)
    except:
        print("err: File is not loading correctly!")
        return False
    print("successful.")

def round_sig_down(x, n):
    return round(x, n - int(math.floor(math.log10(abs(x)))) - 1)

def round_sig_up(x, n):
    return round(x, n - int(math.ceil(math.log10(abs(x)))) - 1)

def save_mixed_plot(folder_path, save_name, data_list, spectrum_range, stacked_data):
    
    fig = go.Figure()
    
    filename_html = "interactive_stack.html"
    
    plot_folder = os.path.join(folder_path, save_name)
    html_path = os.path.join(plot_folder, filename_html)
    
    for i, df in enumerate(data_list):
        fig.add_trace(go.Scatter(
            x=df["wavelength"],
            y=df["flux"],
            mode="lines",
            name=f"{spectrum_range['start_value'][i]} to {spectrum_range['end_value'][i]}Spectrum {i+1}"
        ))
        
    fig.add_trace(go.Scatter(
        x=stacked_data["wavelength"],
        y=stacked_data["flux"],
        mode="lines",
        name="Stacked spectrum"
        ))
    
    fig.update_layout(
        title="Interaktives Stacking / Overlay",
        xaxis_title="Wavelength [Å]",
        yaxis_title="Flux",
        hovermode="x unified"
    )
    
    fig.write_html(html_path)
    print(f"Interactive mixed spectra is exported to {html_path}.")
    
    return plot_folder
    

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
    
    spectrum_range = pd.DataFrame({
        "start_value": start_values,
        "end_value": end_values
        })
    
    for i in range(len(spectrum_range)):
        spectrum_range.loc[i, "start_value"] = round_sig_down(spectrum_range["start_value"][i], 4)
        
    for i in range(len(spectrum_range)):
        spectrum_range.loc[i, "end_value"] = round_sig_up(spectrum_range["end_value"][i], 4)
    
    return max_step, min_length, max_length, spectrum_range



def stacking(file_path, specUnit, fluxUnit, binfactor, z=1, statistic="mean", single_plotting=False, save_name=False): #", min_Res = 10000" if you want/can check the resolution
    
    if not save_name:
        save_name = time.strftime("%Y%m%d-%H%M%S")
    
    fluxcon = FluxConservingResampler()
    folder_path = os.getcwd()

    data_df = []
    id_counter = 1
    temp_data = []
    mixed_spec_list = []

    spec_unit = u.Unit(specUnit)
    flux_unit = u.Unit(fluxUnit)

    for data in file_path:
        
        print(f"File {id_counter} is being processed!")
        check_file(data)
        
        fits_name = os.path.basename(data).replace(".fits", "")
        
        with fits.open(data) as f:
            specdata = f[0].data
            crval1 = f[0].header['CRVAL1']
            cdelt1 = f[0].header['CDELT1']
            
            # spaltweite
            
            
            entries = len(specdata)

        # Calculate resolution 
        """ 
        slitwidth = 
        Res = crval1 / slitwidth
        if Res < min_Res:
            print(f"The resolution of spectrum {os.path} is too small! It will not be used in the stack!")
            pass """
        
        
        
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
        

        if single_plotting:
            
            temp_data = pd.DataFrame({
                        "wavelength": wavelength.value,
                        "flux": flux.value
                        })
            
            png_path, html_path = save_plot(folder_path, save_name, fits_name, id_counter, temp_data)
            
            print(f"PNG Figure saved under path: {png_path}")
            print(f"HTML Figure saved under path: {html_path}")


        # Create Spectrum1D
        input_spec = Spectrum(
            spectral_axis=wavelength,
            flux=flux
        )

        # New dispersion grid
        
        step_length, min_length, max_length, spectrum_range = wavelength_values(file_path)
        
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
        mixed_spec_list.append(df)
        
        id_counter += 1

    # Combine all spectra
    combined = pd.concat(data_df, ignore_index=True)

    # Sort by wavelength
    combined.sort_values("wavelength", inplace=True)

    # Stack (mean per wavelength)
    if statistic == "mean":
        stacked = combined.groupby("wavelength").mean().reset_index()
    elif statistic == "median":
        stacked = combined.groupby("wavelength").median().reset_index()
        
    stacked_save = save_plot(folder_path, save_name, "stacked", "stacked", stacked)
    
    save_folder = save_mixed_plot(folder_path, save_name, mixed_spec_list, spectrum_range, stacked)
    
    print(f"Max steplength from all spectra is: {step_length}")
    print(f"Stacked image saved under path: {stacked_save}")
    
    # Convert back to quantities
    stacked["wavelength"] = stacked["wavelength"] * spec_unit
    stacked["flux"] = stacked["flux"] * flux_unit

    save_as_fits(folder_path, save_name, stacked["flux"],stacked["wavelength"][1]-stacked["wavelength"][0],stacked["wavelength"][0])
    
    return save_folder

def save_as_fits(folder_path, save_name, flux, delta, first_wv):
    plot_folder = os.path.join(folder_path, save_name)
    os.makedirs(plot_folder, exist_ok=True)

    file_path = os.path.join(plot_folder, "stacked.fits")

    # Create 1D flux / errors
    hdu = fits.PrimaryHDU(data=flux)

    print("--- Created new HDUs.")

    # Save header data
    header = hdu.header
    header['OBJECT'] = 'stacked'
    header['CUNIT1'] = 'log'
    header['CRPIX1'] = 1
    header['CRVAL1'] = first_wv
    header['CDELT1'] = delta
    header['CRDER1'] = 0
    header['CTYPE1'] = 'WAVELENGTH'

    print("--- Saved headers.")

    hdu.writeto(file_path, overwrite=True)

    print(f"--- File {file_path} saved.")

# Main
def main():
    app = FileSizeApp()
    #app.mainloop()

# Main
if __name__ == "__main__":
    main() 