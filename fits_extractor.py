"""
© 2025 Julius Richard Dreisbach – FITS Extractor Utility
Designed for rapid spectrum extraction and normalization.

You may use, copy, and modify this software for personal or educational purposes.
Commercial use is not allowed without permission from the author.
No warranty is provided.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from astropy.io import fits
import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import string
import random

version = "1.0"

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

class FileSizeApp(TkinterDnD.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"FITS Extractor {version}")
        self.geometry("550x300")
        self.dnd_text = "Drag & Drop the .fits file or directory here"
        self.all_loaded_files = []
        self.drop_disabled = False
        self.key_values = [wave_key, flux_key, err_key, continuum_key, status_key]
        self._build_ui()

    def _build_ui(self):

        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side="left", padx=10, pady=10)

        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side="right", padx=10, pady=10)

        ttk.Label(self.left_frame, text="How To Use: (1) Provide a file or directory (2) Load key values\n(3) Select the correct key values (4) Extract .fits spectra", font=("Arial", 9)).pack(pady=5)

        self.drop_frame = ttk.Frame(self.left_frame, width=350, height=100, relief="solid", borderwidth=1)
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

        self.options_frame = ttk.Frame(self.left_frame)
        self.options_frame.pack(pady=5)

        self.do_normalize = tk.BooleanVar(value=True)
        self.do_normalize_checkbox = ttk.Checkbutton(self.options_frame, text="Perform Normalization", variable=self.do_normalize)
        self.do_normalize_checkbox.pack(side="left", padx=(0, 10))

        self.do_plot = tk.BooleanVar(value=False)
        self.show_plot_checkbox = ttk.Checkbutton(self.options_frame, text="Plot Extracted Spectra", variable=self.do_plot)
        self.show_plot_checkbox.pack(side="left", padx=(0, 10))

        # GUI frame for spectra extraction
        self.extraction_frame = ttk.Frame(self.left_frame)
        self.extraction_frame.pack(pady=5)

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
        """Wird aufgerufen, wenn eine Datei oder ein Ordner ins Feld gezogen wird."""
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
        #print(self.key_values)
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
            extr_status = create_spectrum(file, key_values, do_cont_extract=self.use_continuum_key.get(), do_status_extract=self.use_status_key.get(), do_normalize=self.do_normalize.get(), show_plot=self.do_plot.get())
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

def normalize(flux_values, flux_err_values):
    """
    How this works: Generate a maximum flux value as median from a small window around the actual flux maximum.
    This prevents the maximum to be without any error value when normalized.
    """

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

def create_spectrum(file_path, key_values, do_cont_extract = False, do_status_extract = False, do_normalize = False, show_plot = True):
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

    if show_plot:
        plt.figure(figsize=(8,5))
        plt.errorbar(wave_flat, flux_flat, yerr=err_flat, fmt='none', ecolor='#33FF33')
        plt.plot(wave_flat, flux_flat, 'b')
        plt.title("Data Plot")
        plt.xlabel(f"Wavelength ({wv_unit})")
        plt.ylabel("Flux")

    print("--- Filtered data.")

    delta_wave = np.diff(wave_flat)
    mean_delta = np.mean(delta_wave)
    std_delta = np.std(delta_wave)
    rel_std = std_delta / mean_delta

    #print(f"[INFO] mean_delta: {mean_delta} {wv_unit}; std_delta: {std_delta} {wv_unit}; rel_std: {rel_std*100} %")

    new_wave = np.linspace(wave_flat.min(), wave_flat.max(), len(wave_flat))
    flux_interp = interp1d(wave_flat, flux_flat, kind='linear', fill_value='extrapolate')
    err_interp = interp1d(wave_flat, err_flat, kind='linear', fill_value='extrapolate')
    new_flux = flux_interp(new_wave)
    new_err = err_interp(new_wave)
    new_wv_delta = new_wave[1] - new_wave[0]

    #print(f"[INFO] wavelength_delta (interpolated): {new_wv_delta} {wv_unit}")

    print("--- Interpolated wavelengths.")

    if do_normalize:
        norm_flux, norm_err, norm_status = normalize(new_flux, new_err)
        match norm_status:
            case 0:
                print("--- Normalized flux values.")
            case 1:
                print("-!- Flux normalization cancelled (unknown error) [error code 1]")
            case 2:
                print("-!- Flux normalization cancelled: Flux is negative. [error code 2]") 
    else:
        norm_flux = new_flux
        norm_err = new_err

    # Create 1D flux / errors
    hdu = fits.PrimaryHDU(data=norm_flux)
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

    header_err = hdu_err.header
    header_err['OBJECT'] = obj_name
    header_err['CUNIT1'] = wv_unit
    header_err['CRPIX1'] = 1
    header_err['CRVAL1'] = new_wave[0]
    header_err['CDELT1'] = new_wv_delta
    header_err['CRDER1'] = std_delta
    header_err['CTYPE1'] = 'WAVELENGTH'

    print("--- Saved headers.")

    id = id_generator(5, "abcdefghik123456")

    interp_spec = "_interp"
    interp_err = "_interp"

    if do_status_extract:
        interp_spec += "_stat"
        interp_err += "_stat"
    if do_normalize:
        interp_spec += f"_norm{norm_status}"
        interp_err += f"_norm{norm_status}"

    interp_spec += "_spec.fits"
    interp_err += "_err.fits"

    new_filename = folder_name + "/" + filename + "_" + obj_name + "_" + id + interp_spec
    new_filename_err = folder_name + "/" + filename + "_" + obj_name + "_" + id + interp_err

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    hdu.writeto(new_filename, overwrite=True)
    hdu_err.writeto(new_filename_err, overwrite=True)

    print(f"--- Files {new_filename}, {new_filename_err} created (Generated ID: {id}).")

    if show_plot:
        plt.show()

    return True

# Main
if __name__ == "__main__":

    app = FileSizeApp()
    app.mainloop()      