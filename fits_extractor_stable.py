"""
© 2025 Julius Richard Dreisbach – FITS Extractor Utility
Designed for rapid spectrum extraction and normalization.

You may use, copy, and modify this software for personal or educational purposes.
Commercial use is not allowed without permission from the author.
No warranty is provided.
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from astropy.io import fits
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import string
import random
import time

version = "1.4.1"

obj_key = 'OBJECT'
wv_unit_key = 'TUNIT1'
wave_key = 'WAVE'
flux_key = 'FLUX'
err_key = 'ERR_FLUX'
continuum_key = 'CONTINUUM'
status_key = 'STATUS'

show_maximum_files = 25

# Set a folder to save the files into.
folder_name = "extracted" + time.strftime("%Y%m%d-%H%M%S")

class FileSizeApp(TkinterDnD.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"FITS Extractor {version}")
        self.geometry("585x320")
        self.dnd_text = "Drag & Drop the .fits file or directory here"
        self.all_loaded_files = []
        self.drop_disabled = False
        self.key_values = [wave_key, flux_key, err_key, continuum_key, status_key]
        self._build_ui()

    def _build_ui(self):

        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side="left", padx=10, pady=10)

        self.right_frame = tk.Frame(self, borderwidth=1, relief=RIDGE)
        self.right_frame.pack(side="right", padx=10, pady=10)

        ttk.Label(self.left_frame, text="How To Use: (1) Provide a file or directory (2) Load key values\n(3) Select the correct key values (4) Extract .fits spectra", font=("Arial", 9)).pack(pady=5)

        self.drop_frame = tk.Frame(self.left_frame, width=400, height=100, relief="solid", borderwidth=1)
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
        self.file_frame.pack(pady=2)

        # label for file amount
        self.file_label = ttk.Label(self.file_frame, text="Files loaded: none", font=("Arial", 11))
        self.file_label.pack(side="left", padx=(0, 10))

        # label for file size
        self.result_label = ttk.Label(self.file_frame, text="", font=("Arial", 11))
        self.result_label.pack(side="left", padx=(0, 10))

        # show all files button
        self.show_all_files_button = ttk.Button(self.file_frame, text="Show Files", command=self.show_all_files)
        self.show_all_files_button.pack(side="right")

        # Comboboxes for bintable columns
        self.check_bintable_button = ttk.Button(self.right_frame, text="Load Key Values", state=tk.DISABLED, command=self.check_bintable_keys)
        self.check_bintable_button.pack(pady=5)

        self.cb_wave = ttk.Combobox(self.right_frame,state=tk.DISABLED)
        self.cb_wave.set("Select wavelength key...")
        self.cb_wave.pack(pady=5)

        self.cb_flux = ttk.Combobox(self.right_frame,state=tk.DISABLED)
        self.cb_flux.set("Select flux key...")
        self.cb_flux.pack(pady=5)

        self.cb_flux_err = ttk.Combobox(self.right_frame,state=tk.DISABLED)
        self.cb_flux_err.set("Select flux error key...")
        self.cb_flux_err.pack(pady=5)

        self.cb_cont = ttk.Combobox(self.right_frame,state=tk.DISABLED)
        self.cb_cont.set("Select continuum key...")
        self.cb_cont.pack(pady=5)

        self.use_continuum_key = tk.BooleanVar(value=False)
        self.use_continuum_checkbox = ttk.Checkbutton(self.right_frame, text="Use Continuum Key", variable=self.use_continuum_key, state=tk.DISABLED, command=self.toggle_combobox_activation)
        self.use_continuum_checkbox.pack(pady=5)

        self.cb_status = ttk.Combobox(self.right_frame,state=tk.DISABLED)
        self.cb_status.set("Select status key...")
        self.cb_status.pack(pady=5)

        self.use_status_key = tk.BooleanVar(value=False)
        self.use_status_checkbox = ttk.Checkbutton(self.right_frame, text="Use Status Key", variable=self.use_status_key, state=tk.DISABLED, command=self.toggle_combobox_activation)
        self.use_status_checkbox.pack(pady=5)

        self.options_frame = tk.Frame(self.left_frame, relief="flat", borderwidth=1)
        self.options_frame.pack(pady=2)

        self.do_interpolate = tk.BooleanVar(value=True)
        self.do_interpolate_checkbox = ttk.Checkbutton(
        self.options_frame, text="Interpolate Values", variable=self.do_interpolate)
        self.do_interpolate_checkbox.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.do_cont_subtraction = tk.BooleanVar(value=True)
        self.do_cont_subtraction_checkbox = ttk.Checkbutton(
        self.options_frame, text="Subtract Continuum", variable=self.do_cont_subtraction)
        self.do_cont_subtraction_checkbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.do_normalize = tk.BooleanVar(value=False)
        self.do_normalize_checkbox = ttk.Checkbutton(
        self.options_frame, text="Perform Normalization", variable=self.do_normalize)
        self.do_normalize_checkbox.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.do_sn_calc = tk.BooleanVar(value=False)
        self.do_sn_calc_checkbox = ttk.Checkbutton(
        self.options_frame, text="Calculate S/N Values", variable=self.do_sn_calc)
        self.do_sn_calc_checkbox.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # GUI frame for spectra extraction
        self.extraction_frame = ttk.Frame(self.left_frame)
        self.extraction_frame.pack(pady=5)

        self.do_plot = tk.BooleanVar(value=False)
        self.show_plot_checkbox = ttk.Checkbutton(self.extraction_frame, text="Plot Extracted Spectra", variable=self.do_plot)
        self.show_plot_checkbox.pack(side="left", padx=(0, 10))

        # extraction button
        self.extraction_button = ttk.Button(self.extraction_frame, text="Extract .fits Spectra", state=tk.DISABLED, command=self.extract_spectra)
        self.extraction_button.pack(side="left", padx=(0, 10))

        self.progress_bar = ttk.Progressbar(self.extraction_frame)
        self.progress_bar.pack(side="right")

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
        self.check_bintable_button.config(state=tk.NORMAL)

    def cb_get_values(self):
        self.key_values[0] = self.cb_wave.get()
        self.key_values[1] = self.cb_flux.get()
        self.key_values[2] = self.cb_flux_err.get()
        self.key_values[3] = self.cb_cont.get()
        self.key_values[4] = self.cb_status.get()
        return self.key_values

    def check_bintable_keys(self):
        if self.all_loaded_files == []:
            tk.messagebox.showwarning(title="Not possible", message="Cannot load key values, as no files are loaded.\n\nPlease provide a file or directory first.")
            return
        
        self.extraction_button.config(state=tk.DISABLED)
        self.check_bintable_button.config(state=tk.DISABLED)

        self.possible_keys = []
        for file in self.all_loaded_files:
            self.update_idletasks()
            keys = check_file(file)
            if not keys == False:
                for key in keys:
                    if key not in self.possible_keys:
                        self.possible_keys.append(key)

        self.update_comboboxes(self.possible_keys)

        self.extraction_button.config(state=tk.NORMAL)
            
    def update_comboboxes(self, keys):
        self.cb_wave.config(values=keys,state='readonly')
        self.cb_flux.config(values=keys,state='readonly')
        self.cb_flux_err.config(values=keys,state='readonly')
        self.cb_cont.config(values=keys)
        self.cb_status.config(values=keys)
        self.use_continuum_checkbox.config(state=tk.NORMAL)
        self.use_status_checkbox.config(state=tk.NORMAL)
        self.toggle_combobox_activation()

    def toggle_combobox_activation(self):
        if self.use_continuum_key.get():
            self.cb_cont.config(state='readonly')
        else:
            self.cb_cont.config(state=tk.DISABLED)

        if self.use_status_key.get():
            self.cb_status.config(state='readonly')
        else:
            self.cb_status.config(state=tk.DISABLED)

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

    def progress_bar_step(self):
        self.progress_bar.step()
        self.update_idletasks()

        current_value = self.progress_bar['value']
        maximum_value = self.progress_bar['maximum']

        if current_value >= maximum_value:
            pass
            #self.on_closing(self)

    def extract_spectra(self):
        if self.all_loaded_files == []:
            tk.messagebox.showwarning(title="Not possible", message="Cannot begin extraction, as no files are loaded.\n\nPlease provide a file or directory first.")
            return

        self.drop_disabled = True
        self.drop_label.config(text="Drag & Drop disabled")
        self.show_plot_checkbox.config(state=tk.DISABLED)
        self.extraction_button.config(state=tk.DISABLED)
        self.progress_bar.config(maximum=len(self.all_loaded_files))
        #print(self.do_plot.get())
        key_values = self.cb_get_values()

        i = 1
        extr = 0
        skip = 0
        for file in self.all_loaded_files:
            #thread = threading.Thread(target=create_spectrum, args=(file, self.do_plot.get()), daemon=True).start()
            print(f"--###-- {i}/{len(self.all_loaded_files)} --###--")
            extr_status = create_spectrum(file, key_values, do_cont_extract=self.use_continuum_key.get(), do_status_extract=self.use_status_key.get(), do_interpolate=self.do_interpolate.get(), 
                                          do_continuum_removal = self.do_cont_subtraction.get(), do_normalize=self.do_normalize.get(), do_sn_calc=self.do_sn_calc.get(), show_plot=self.do_plot.get())
            if extr_status:
                extr += 1
            else:
                skip += 1
            self.progress_bar_step()
            i+=1

        print(f"--###-- FINISHED --###--")
        print(f"Attempted to extract spectra from {len(self.all_loaded_files)} files")
        print(f"   - successful for {extr} files")
        print(f"   - skipped {skip} files")
        if skip > 0:
            tk.messagebox.showinfo(title="Success", message="Successfully extracted spectra of the provided files.", detail=f"Please note that {skip} file(s) have been skipped in the process as they are incompatible for various reasons. See the console for more information on these files.")
        else:
            tk.messagebox.showinfo(title="Success", message="Successfully extracted spectra of the provided files.")
        self.on_closing()

    def on_closing(self):
        self.destroy()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def normalize_max(flux_values, flux_err_values):
    """
    How this works: Generate a maximum flux value as median from a small window around the actual flux maximum.
    This prevents the maximum to be without any error value when normalized.
    """
    try:
        imax = np.argmax(flux_values)
        win = 5  # window half-width for maximum determination
        i0, i1 = max(0, imax-win), min(len(flux_values), imax+win+1)
        f_max = np.median(flux_values[i0:i1])

        if f_max < 0:
            return flux_values, flux_err_values, 2

        err_noise = np.median(flux_err_values)
        err_f_max = err_noise / np.sqrt(i1-i0)

        # normalize flux and error values
        flux_norm = flux_values / f_max
        err_norm = np.sqrt((flux_err_values / f_max)**2 + (flux_values * err_f_max / f_max**2)**2)

        return flux_norm, err_norm, 0
    except Exception as e:
        print(f"Unknown error: {e}")
        return flux_values, flux_err_values, 1

# Currently not in use.
def subtract_continuum(flux, cont):
    flx_len = len(flux)
    cnt_len = len(cont)

    flux_sub = []
    if flx_len == cnt_len:
        for i in range(flx_len):
            flux_sub.append(flux[i]-cont[i])
    else:
        return False # cont & flux differ in length

    return np.array(flux_sub)

# Currently not in use.
def savgol_smooth(flux, window=51, poly=3, err=None):
    """
    Use Savitzky-Golay filtering to smooth spectrum. 
    
    Parameters
    ---------
    flux : array
        raw flux values
    window : int, optional
        smoothing window size (odd parity)
    poly : int, optional
        polynomial degree
    err : array_like or None
        optional flux error values
    
    Returns
    --------
    smooth_flux : ndarray
        Smoothened flux values
    smooth_err : ndarray or None
        Smoothened flux error values (None if none given)
    """
    # Make window size odd if needed
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 3 - (poly % 2)  # Set minimum size
    
    smooth_flux = savgol_filter(flux, window_length=window, polyorder=poly)

    if err is None:
        return smooth_flux, None
    else:
        smooth_err = savgol_filter(err, window_length=window, polyorder=poly)
        return smooth_flux, smooth_err
     
def estimate_continuum_sg(wave, flux, window_length=401, polyorder=3, med_width=201, peak_sigma=5.0, mask_width_pix=50, iter_clip=True, niter=2):
    """
    Estimate the spectral continuum using an iterative masked Savitzky-Golay filtering
    approach. Strong spectral lines are identified via robust sigma clipping, masked,
    interpolated over, and the continuum is then estimated from the smoothed spectrum.

    Parameters
    ----------
    wave : array-like
        Wavelength values of the spectrum.
    flux : array-like
        Flux values corresponding to `wave`.
    window_length : int, optional
        Savitzky-Golay smoothing window size (must be odd). Default is 401.
    polyorder : int, optional
        Polynomial order for the Savitzky-Golay filter. Default is 3.
    med_width : int, optional
        Window size of the median filter used for initial peak detection.
        Default is 201.
    peak_sigma : float, optional
        Sigma threshold above the median-filtered background for identifying
        significant peaks to mask. Default is 5.0.
    mask_width_pix : int, optional
        Half-width of the region to mask around every detected peak (in pixels).
        Default is 50.
    iter_clip : bool, optional
        If True, iteratively re-detect and expand masks using the updated continuum
        estimate. Default is True.
    niter : int, optional
        Maximum number of sigma-clipping iterations. Only relevant if 'iter_clip'
        is True. Default is 2.

    Returns
    -------
    continuum : ndarray
        The estimated continuum across the wavelength grid.
    flux_contsub : ndarray
        Flux values with the estimated continuum subtracted ('flux - continuum').
    mask : ndarray of bool
        Boolean mask array; True indicates masked (peak) data points.
    parameters : list
        List of the parameters used for the continuum estimation in the order:
        [window_length, polyorder, med_width, peak_sigma, mask_width_pix, niter]

    Notes
    -----
    The algorithm proceeds as follows:
    1. A median filter provides a coarse background estimate.
    2. Residuals are compared against a robust sigma threshold (MAD-based).
    3. Identified peaks are masked, including ±'mask_width_pix' around each peak.
    4. The spectrum is linearly interpolated over the masked regions.
    5. A Savitzky-Golay filter estimates the continuum from the interpolated data.
    6. If `iter_clip` is enabled, steps 3-5 are repeated up to 'niter' times.

    This method is effective for spectra containing narrow absorption/emission
    features superimposed on a slowly varying continuum.
    """
    if window_length % 2 == 0:
        window_length += 1
    x = np.asarray(wave, dtype=float)
    y = np.asarray(flux, dtype=float)
    # 1. A median filter provides a coarse background estimate.
    y_med = median_filter(y, size=med_width, mode='nearest')

    # 2. Residuals are compared against a robust sigma threshold (MAD-based).
    resid = y - y_med
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad == 0:
        mad = np.std(resid) + 1e-12

    #3. Identified peaks are masked, including ±'mask_width_pix' around each peak.
    threshold = np.median(resid) + peak_sigma * 1.4826 * mad
    peaks = resid > threshold

    # grow mask around peaks
    mask = np.zeros_like(y, dtype=bool)
    idxs = np.where(peaks)[0]
    for i in idxs:
        lo = max(0, i - mask_width_pix)
        hi = min(len(y), i + mask_width_pix + 1)
        mask[lo:hi] = True

    # optional iterative widening (to catch lines moving after subtract)
    for it in range(niter if iter_clip else 1):
        # 4. The spectrum is linearly interpolated over the masked regions.
        xi = x[~mask]
        yi = y[~mask]
        # If too few points, break
        if yi.size < max(10, polyorder+2):
            # fallback: apply SG to original (best-effort)
            cont = savgol_filter(y, window_length, polyorder)
            return cont, y - cont, mask
        interp = np.interp(x, xi, yi)
        # 5. A Savitzky-Golay filter estimates the continuum from the interpolated data.
        cont = savgol_filter(interp, window_length, polyorder)
        # 6. If `iter_clip` is enabled, steps 3-5 are repeated up to 'niter' times.
        resid2 = y - cont
        mad2 = np.median(np.abs(resid2 - np.median(resid2)))
        threshold2 = np.median(resid2) + peak_sigma * 1.4826 * (mad2 if mad2>0 else 1e-12)
        new_peaks = resid2 > threshold2
        if not new_peaks.any() or not iter_clip:
            break
        idxs = np.where(new_peaks)[0]
        for i in idxs:
            lo = max(0, i - mask_width_pix)
            hi = min(len(y), i + mask_width_pix + 1)
            mask[lo:hi] = True

    return cont, y - cont, mask, [window_length, polyorder, med_width, peak_sigma, mask_width_pix, niter]

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

    try:
        head = hdul[0].header # Header
        table_head = hdul[1].header # Table Header
        table_columns = hdul[1].columns.names # BinTable Column names

        return table_columns
    except:
        print("err: File throws error when extracting!")
        return False

def create_spectrum(file_path, key_values, do_cont_extract = False, do_status_extract = False, do_interpolate = False, do_continuum_removal = False, do_normalize = False, do_sn_calc = False, show_plot = True):
    print(f"Now extracting: {file_path}", end = ' ')

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

    try:
        head = hdul[0].header # Header
        table_head = hdul[1].header # Table Header
        table_data = hdul[1].data  # BinTable


        obj_name = head[obj_key]
        wv_unit = table_head[wv_unit_key]

        # Extract data columns
        wave = table_data[key_values[0]]
        flux = table_data[key_values[1]]
        err = table_data[key_values[2]]
        if do_cont_extract:
            cont = table_data[key_values[3]]

        if do_status_extract:
            status = table_data[key_values[4]]
    except:
        print("err: File throws error when extracting!")
        return False

    print("--- Extracted data.")

    if do_status_extract:
        # Filter by status int
        filtered_flux = np.where(status == 1, flux, 0)
        filtered_err = np.where(status == 1, err, 0)
    else:
        filtered_flux = flux
        filtered_err = err

    # Flatten arrays
    wave_flat = wave[0].flatten()
    flux_flat = filtered_flux[0].flatten()
    err_flat = filtered_err[0].flatten()
    if do_cont_extract:
        cont_flat = cont[0].flatten()

    print("--- Filtered data.")

    delta_wave = np.diff(wave_flat)
    mean_delta = np.mean(delta_wave)
    std_delta = np.std(delta_wave)
    rel_std = std_delta / mean_delta

    #print(f"[INFO] mean_delta: {mean_delta} {wv_unit}; std_delta: {std_delta} {wv_unit}; rel_std: {rel_std*100} %")

    if do_interpolate:
        new_wave = np.linspace(wave_flat.min(), wave_flat.max(), len(wave_flat))
        flux_interp = interp1d(wave_flat, flux_flat, kind='linear', fill_value='extrapolate')
        err_interp = interp1d(wave_flat, err_flat, kind='linear', fill_value='extrapolate')
        if do_cont_extract:
            cont_interp = interp1d(wave_flat, cont_flat, kind='linear', fill_value='extrapolate')
            new_cont = cont_interp(new_wave)
        new_flux = flux_interp(new_wave)
        new_err = err_interp(new_wave)
        new_wv_delta = new_wave[1] - new_wave[0]
        print("--- Interpolated wavelengths.")
    else:
        # Note that this is not exact, in order to be able to construct this data without binary tables the wavelength delta needs to be constant over the whole spectrum, essentially shifting 
        # flux values to new wavelengths without correcting for error (do_interpolate does that). It is recommended to activate interpolation if the wavelength delta is not constant beforehand.
        new_wave = wave_flat
        new_flux = flux_flat
        new_err = err_flat
        if do_cont_extract:
            new_cont = cont_flat
        new_wv_delta = (new_wave[-1] - new_wave[0]) / (len(new_wave) - 1)

    if do_continuum_removal:
        it_smooth_flux, cont_flux, _, params = estimate_continuum_sg(new_wave,new_flux,window_length=601, polyorder=3,med_width=401, peak_sigma=5.0,mask_width_pix=60, iter_clip=True, niter=3)
        cont_err = new_err
        # Deactivated until further notice
        """
        smooth_flux, _ = savgol_smooth(new_flux,round(len(new_flux)/15),3,new_err)
        if do_cont_extract:
            cont_flux = subtract_continuum(new_flux, new_cont)
        else:
            cont_flux = subtract_continuum(new_flux, smooth_flux)
        cont_err = new_err
        """
        print("--- Fitted and subtracted continuum.")
    else:
        cont_flux = new_flux
        cont_err = new_err

    if do_normalize:
        norm_flux, norm_err, norm_status = normalize_max(cont_flux, cont_err)
        match norm_status:
            case 0:
                print("--- Normalized flux values with respect to maximum flux.")
            case 1:
                print("-!- Flux normalization cancelled (unknown error) [error code 1]")
            case 2:
                print("-!- Flux normalization cancelled: Flux is negative. [error code 2]") 
    else:
        norm_flux = cont_flux
        norm_err = cont_err

    if do_sn_calc:
        sn_vals = norm_flux / norm_err
        print("--- Calculated S/N values.")
    else:
        sn_vals = [0] * norm_flux

    # Create 1D flux / errors
    hdu = fits.PrimaryHDU(data=norm_flux)
    hdu_sn = fits.PrimaryHDU(data=sn_vals)
    hdu_err = fits.PrimaryHDU(data=norm_err)

    print("--- Created new HDUs.")

    # Save header data
    header = hdu.header
    header['OBJECT'] = obj_name
    header['CUNIT1'] = wv_unit
    header['CRPIX1'] = 1
    header['CRVAL1'] = new_wave[0]
    header['CDELT1'] = new_wv_delta
    header['CRDER1'] = std_delta
    header['CTYPE1'] = 'WAVELENGTH'
    header['CONTSUB'] = do_continuum_removal
    header.comments['CONTSUB'] = 'Iterative continuum subtraction'
    if do_continuum_removal:
        header['PWINLEN'] = params[0]
        header.comments['PWINLEN'] = 'contsub window_length parameter'
        header['PPOL_ORD'] = params[1]
        header.comments['PPOL_ORD'] = 'contsub polyorder parameter'
        header['PMEDWID'] = params[2]
        header.comments['PMEDWID'] = 'contsub med_width parameter'
        header['PPEAKSIG'] = params[3]
        header.comments['PPEAKSIG'] = 'contsub peak_sigma parameter'
        header['PMASKWID'] = params[4]
        header.comments['PMASKWID'] = 'contsub mask_width_pix parameter'
        header['PNITER'] = params[5]
        header.comments['PNITER'] = 'contsub number of iterations'

    header_err = hdu_err.header
    header_err['OBJECT'] = obj_name
    header_err['CUNIT1'] = wv_unit
    header_err['CRPIX1'] = 1
    header_err['CRVAL1'] = new_wave[0]
    header_err['CDELT1'] = new_wv_delta
    header_err['CRDER1'] = std_delta
    header_err['CTYPE1'] = 'WAVELENGTH'
    header_err['CONTSUB'] = do_continuum_removal
    header_err.comments['CONTSUB'] = 'Iterative continuum subtraction'
    if do_continuum_removal:
        header_err['PWINLEN'] = params[0]
        header_err.comments['PWINLEN'] = 'contsub window_length parameter'
        header_err['PPOL_ORD'] = params[1]
        header_err.comments['PPOL_ORD'] = 'contsub polyorder parameter'
        header_err['PMEDWID'] = params[2]
        header_err.comments['PMEDWID'] = 'contsub med_width parameter'
        header_err['PPEAKSIG'] = params[3]
        header_err.comments['PPEAKSIG'] = 'contsub peak_sigma parameter'
        header_err['PMASKWID'] = params[4]
        header_err.comments['PMASKWID'] = 'contsub mask_width_pix parameter'
        header_err['PNITER'] = params[5]
        header_err.comments['PNITER'] = 'contsub number of iterations'

    hdu_sn.header = header

    print("--- Saved headers.")

    id = id_generator(5, "abcdefghik123456")

    final = ""

    if do_interpolate:
        final += "_interp"
    if do_status_extract:
        final += "_stat"
    if do_continuum_removal:
        final += "_cntrm"
    if do_normalize:
        final += f"_norm{norm_status}"

    final_spec = final + "_spec.fits"
    final_sn = final + "_sn.fits"
    final_err = final + "_err.fits"

    new_filename = folder_name + "/" + filename + "_" + obj_name + "_" + id + final_spec
    new_filename_sn = folder_name + "/" + filename + "_" + obj_name + "_" + id + final_sn
    new_filename_err = folder_name + "/" + filename + "_" + obj_name + "_" + id + final_err

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    hdu.writeto(new_filename, overwrite=True)
    hdu_err.writeto(new_filename_err, overwrite=True)

    if do_sn_calc:
        hdu_sn.writeto(new_filename_sn, overwrite=True)
        print(f"--- Files {new_filename}, {new_filename_sn}, {new_filename_err} created (Generated ID: {id}).")
    else:
        print(f"--- Files {new_filename}, {new_filename_err} created (Generated ID: {id}).")

    if show_plot:
        plt.figure(figsize=(8,5))
        plt.errorbar(new_wave, norm_flux, yerr=norm_err, fmt='none', ecolor='#33FF33')
        plt.plot(new_wave, norm_flux, 'b')
        plt.title("Data Plot")
        plt.xlabel(f"Wavelength ({wv_unit})")
        plt.ylabel("Flux")
        plt.show()

    return True

# Main
if __name__ == "__main__":

    app = FileSizeApp()
    app.mainloop()      