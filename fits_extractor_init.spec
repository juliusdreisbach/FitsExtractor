# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# Hier sammeln wir alle Daten, Binaries und Hidden Imports für die Problem-Pakete
tcl_data = collect_all('tkinterdnd2')
asdf_data = collect_all('asdf')
astropy_data = collect_all('astropy')
specutils_data = collect_all('specutils')
pandas_data = collect_all('pandas')

a = Analysis(
    ['fits_extractor_init.py'],
    pathex=[],
    binaries=tcl_data[1] + asdf_data[1] + astropy_data[1] + specutils_data[1] + pandas_data[1],
    datas=tcl_data[0] + asdf_data[0] + astropy_data[0] + specutils_data[0] + pandas_data[0],
    hiddenimports=[
        'scipy._cyutility', 
        'scipy.special._cdflib', 
        'scipy.linalg._cythonized_array_utils'
    ] + tcl_data[2] + asdf_data[2] + astropy_data[2] + specutils_data[2] + pandas_data[2],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fits_extractor_init',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
