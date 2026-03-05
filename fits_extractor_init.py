import tkinter as tk
from tkinter import *
from tkinter import ttk
import subprocess
import sys

def run_fits_extractor_old():
    subprocess.Popen([sys.executable, "fits_extractor_v1.4.py"])

def run_fits_extractor_stacker_combi():
    subprocess.Popen([sys.executable, "fits_extractor_stacker.py"])

root = tk.Tk()
root.title("Choose your FITS Extractor version")
root.geometry("540x240")

top_frame = tk.Frame()
top_frame.pack(side="top", padx=2, pady=2)

bottom_frame = tk.Frame()
bottom_frame.pack(side="bottom", padx=2, pady=2)

left_frame = tk.Frame(bottom_frame, borderwidth=1, relief=RIDGE)
left_frame.pack(side="left", padx=5, pady=5)

right_frame = tk.Frame(bottom_frame, borderwidth=1, relief=RIDGE)
right_frame.pack(side="right", padx=5, pady=5)

ttk.Label(top_frame, text="FITS Extractor currently supports two different versions.\nPlease choose which version you want to use.", font=("Arial", 9)).pack(pady=5,fill='both', expand=True)

ttk.Label(left_frame, text="STABLE VERSION v1.4", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
ttk.Label(left_frame, text="This version is identical to the\npreviously released v1.4. The\ndocumentation can be found\non the GitHub page.", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
button_old = tk.Button(left_frame, text="Run FITS Extractor 1.4", command=run_fits_extractor_old)
button_old.pack(padx=20, pady=10)

ttk.Label(right_frame, text="EXPERIMENTAL VERSION v1.5.S", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
ttk.Label(right_frame, text="This version includes a stacking operation\nfor stacking already extracted spectra.\nThis feature is experimental and might\nlead to unexpected results or crashes.\nFor now, no documentation can be provided.", font=("Arial", 9)).pack(padx=20, pady=5,fill='both', expand=True)
button_combi = tk.Button(right_frame, text="Run FITS Extractor 1.5.S (Experimental)", command=run_fits_extractor_stacker_combi)
button_combi.pack(padx=20, pady=10)

root.mainloop()