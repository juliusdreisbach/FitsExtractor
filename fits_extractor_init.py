"""
© 2026 Julius Richard Dreisbach – FITS Extractor Utility
Designed for rapid spectrum extraction and normalization.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinterdnd2 import TkinterDnD
import fits_extractor_base as feb
import fits_extractor_stacker as fes

def run_fits_extractor():
    print("running FITS Extractor ...")
    feb.main()

def run_fits_extractor_stacker():
    print("running FITS Extractor Stacker ...")
    fes.main()

root = TkinterDnD.Tk()
root.title("Choose your FITS Extractor application")
root.geometry("620x200")

top_frame = tk.Frame()
top_frame.pack(side="top", padx=2, pady=2)

bottom_frame = tk.Frame()
bottom_frame.pack(side="bottom", padx=2, pady=2)

left_frame = tk.Frame(bottom_frame, borderwidth=1, relief=RIDGE)
left_frame.pack(side="left", padx=5, pady=5)

right_frame = tk.Frame(bottom_frame, borderwidth=1, relief=RIDGE)
right_frame.pack(side="right", padx=5, pady=5)

ttk.Label(top_frame, text="FITS Extractor consists of two different applications. Please choose which one you want to use.", font=("Arial", 9)).pack(pady=5,fill='both', expand=True)

ttk.Label(left_frame, text="EXTRACTOR v1.4.3", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
ttk.Label(left_frame, text="Extract binary tables from FITS files and create 1D spectra.\nYou may also interpolate or normalize the data.\nAdditional features: Continuum Subtraction, S/N Calculation.\nA documentation can be found on the GitHub page.", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
button_old = tk.Button(left_frame, text="Run FITS Extractor 1.4.3", command=run_fits_extractor)
button_old.pack(padx=20, pady=10)

ttk.Label(right_frame, text="STACKER v1.0.1", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
ttk.Label(right_frame, text="Stack already extracted spectra.\n\nA documentation can be found\non the GitHub page.", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
button_combi = tk.Button(right_frame, text="Run FITS Extractor Stacker 1.0.1", command=run_fits_extractor_stacker)
button_combi.pack(padx=20, pady=10)

root.mainloop()