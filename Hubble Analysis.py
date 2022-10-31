#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:57:54 2020

@author: dawudabd-alghani
"""
# Importing numpy, scipy.optimise and matplotlib.pyplot for later use.
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

# Defining the linear gaussian function which matches the gaussian peak and the straight line fit. 
def linear_gaussian(x, a, mu, std, m, c):
    gaussian = a*np.exp(-(x-mu)**2/(2*std**2))
    linear = m*x+c
    return gaussian + linear

# Calculating the intial guesses of the linear_gaussian function to put into sp.curve_fit.
def initial_guess(x, y):
    m0 = (y[-1]-y[0])/(x[-1]-x[0])
    c0 = y[0] - m0*x[0]
    residuals = y - (m0*x +c0)
    a0 = max(residuals)
    mu0 = x[np.argmax(residuals)]
    std0 = 8e-9
    return [a0, mu0, std0, m0, c0]

# Takes in the x and y values of data and returns the mean and uncertainty in the mean.
# The mean is the observed wavelength so it returns the observed wavelength and its uncertainty.
# It calls the initial_guess function and the linear_gaussian function in order to model the curve.
def observed_wavelength(x, y):
    p0 = initial_guess(x, y)
    params, cov = sp.curve_fit(linear_gaussian, x, y, p0)
    unc = np.sqrt(cov[1,1])
    return params[1], unc

# Takes in the observed wavelength and its uncertainty (which is returned from the observed_wavelength function).
# Returns the recessional velocity (calculated using the redshft formula) in km/s and the uncertainty in velocity.
def velocity(o_w, unc1):
    e_w = 656.28e-9
    c = 2.9979e8
    v = c*(o_w**2 - e_w**2)/(1000*(o_w**2 + e_w**2))
    unc2 = ((c*e_w**2*o_w)/(250*(o_w**2 + e_w**2)**2))*unc1
    return v, unc2

# Creating empty arrays for observations, distances, wavelengths, intensities, velocities and uncertainties to be appended to later.
observations = []
distances = []
wavelengths = []
intensities = []
velocities = []
uncertainties = []

# Reading the Distance_Mpc.csv file and saving the observation numbers of valid responces in the observations array created earlier.
# Saving the distances associated with the observation numbers saved in the distances array created earlier.
with open('Data/Distance_Mpc.csv', 'r') as file:
    for row in file:
        row = row.split(',')
        row[-1] = row[-1].strip()
        if row[2] == '1':
            observations.append(float(row[0]))
            distances.append(float(row[1]))

# Reading the Halpha_spectral_data.csv file and saving the wavelength in the wavelengths array created earlier.
# Saving the intensities associated with the observation numbers in the observations array in the intensities array created earlier.
with open('Data/Halpha_spectral_data.csv', 'r') as file:
    lines = file.readlines()
    for i in lines[2:]:
        wavelengths.append(float(i.split(',')[0]))
    row2 = lines[1]
    row2 = row2.split(',')
    for obs in observations:
        for a in range(1,len(row2)):
            b = row2[a]
            obs_num = b[13:18]
            if float(obs_num) == obs:
                d = []
                for i in lines[2:]:
                    if a == 30:
                        i = i.strip()
                    d.append(float(i.split(',')[a]))
                intensities.append(d)
                break

# For each set of the intensity recordings the velocity is calculated and saved in the velocities array created earlier.
# For each set of the intensity recordings the uncertainty in velocity is calculated and saved in the uncertainties array created earlier.
# The velocities and their uncertainties are calculated by calling the observed_wavelength and velocity functions defined earlier.
for a in intensities:
    o_w, unc1 = observed_wavelength(np.array(wavelengths), np.array(a))
    v, unc2 = velocity(o_w, unc1)
    velocities.append(v)
    uncertainties.append(unc2)

# Creating a plot with a classic style and labelling the axis and title.
plt.style.use('classic')
plt.xlabel('Distance (Mpc)', fontsize=16)
plt.ylabel('Redshift (km/s)', fontsize=16)
plt.title('Determining Hubble\'s constant', fontsize=20)
plt.grid()

# Plotting the distance velocity data points from the distances and velocities arrays.
plt.plot(distances, velocities, 'x')

# Plotting the errorbars due to the uncertainty in the velocity.
# The errorbars are plotted, but too small to be seen.
plt.errorbar(distances, velocities, uncertainties, fmt='none', capsize=4)

# Fitting a straight line to the data and plotting the line.
fit, cov = np.polyfit(distances, velocities, 1, w=1/np.array(uncertainties), cov=1)
f = np.poly1d(fit)
plt.plot(distances, f(distances))

# Saving the plot into a png file.
plt.savefig('Plot of reshift velocity vs distance.png')

# Printing the value of Hubble's constant and its uncertainty.
print('Hubble\'s constant = %.2g ± %.1g km/s/Mpc' %(fit[0], np.sqrt(cov[0,0])))