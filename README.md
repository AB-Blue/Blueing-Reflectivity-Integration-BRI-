# Blueing-Reflectivity-Integration-BRI-
Seismic Spectral Enhancement Technique
proposed by Babasafari et al., 2016
https://library.seg.org/doi/10.1190/ice2016-6513081.1 
--------------------------------------------------------------------------------------------------------------------------------------
# Script files:

bri_class.py: Python script
bri_gui.py: Python script
BRI_2023.ipynb: Jupyter notebook
--------------------------------------------------------------------------------------------------------------------------------------
Author: Amir Abbas Babasafari (AB)
Date: October 2023
Email: a.babasafari@yahoo.com  
--------------------------------------------------------------------------------------------------------------------------------------
# Installation Requirements:

Python 3.9
Libraries: numpy, segyio, matplotlib, scipy, PyQt5
--------------------------------------------------------------------------------------------------------------------------------------
# Run the application:

Download the code from GitHub or clone the repository to your machine
Install the required dependencies using pip install 'library name'
Run the BRI app via command prompt 'python bri_gui.py' in the same directory that bri_class.py and bri_gui.py exist
Or run using any available IDE

--------------------------------------------------------------------------------------------------------------------------------------
The link of tutorial data from open data website:
https://dataunderground.org/dataset/waipuku
--------------------------------------------------------------------------------------------------------------------------------------
# Application features:

Please see description below and for more information please watch the recorded demo 
on LinkedIn and YouTube via provided links below:


load original seismic:                    Load 2D/3D post-stack seismic data in segy/sgy format
save BRI seismic:                         Save new seismic data after BRI application in segy/sgy format
location map:                             Plot XY coordinates map
BRI spectral enhancement:                 Run BRI method 
plot seismic data (original and BRI):     Plot and toggle between original seismic and BRI sections
Amplitude spectrum:                       Plot mean amplitude spectrum of original seismic and BRI 
Overlay comparison:                       A simple overlay comparison between original seismic and BRI traces at selected location
* matplotlib navigation toolbar is available.

--------------------------------------------------------------------------------------------------------------------------------------
# BRI technique:

Blueing Reflectivity Integration (BRI) method aims to produce the same results as seismic spectral blueing, where 
there is not any acoustic impedance log available to create reflection coefficient log. So, by leveraging seismic traces, 
so-called pseudo Reflection Coefficient (RC) are extracted from local maximum or minimum along seismic traces by 
calculating amplitude for zero first derivative. The rest of the steps are similar to Seismic Spectral Blueing.
In addition, the function of creation RC from log is available in BRI main class but not included in BRI GUI for front-end purpose.

--------------------------------------------------------------------------------------------------------------------------------------
Steps:

Calculating mean amplitude spectrum of seismic data reflectivity series in the frequency domain
Obtaining pseudo Reflection Coefficient (RC) derived from seismic data 
Fitting a curve on amplitude spectrum of logarithmic reflection coefficient 
Multiplication of mean amplitude spectrum of seismic data by fitted Blue spectra
Taking inverse Fourier Transformation of BRI spectra to bring back the data to the time domain (BRI operator)
Convolving seismic amplitude with BRI operator 
Quality control steps

--------------------------------------------------------------------------------------------------------------------------------------
# Functions in bri_class:

Function to define data type as 2D or 3D and Post-Stack or Pre-Stack as well as selecting data for display
Function to read 2D/3D post-stack seismic data (segy file) and specify data-related parameters
Function to extract header information of segy file (Geometry-related byte locations)
Function for scatter-plotting of X and Y Geometry
Function to display seismic data (segy file)
Function to calculate amplitude spectrum
Function to calculate mean amplitude spectrum
Function to plot amplitude spectrum
Function to calculate pseudo Reflection Coefficient (RC) from seismic data
Function to compute F4 Index
Function for fitting a curve on RC amplitude spectrum
Function to calculate BRI operator
Function to convolve seismic traces and BRI operator
Function to compare seismic and BRI data at one location
Function to export 2D/3D post-stack BRI data (segy file)
Function for writing header information on segy output (Geometry-related byte locations)
--------------------------------------------------------------------------------------------------------------------------------------
