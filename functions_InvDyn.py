# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:47:00 2024

@author: Kevin
"""
import scipy as sp
import pandas as pd
import numpy as np
#import matplotlip.pyplot as plt

"""Functions"""

def plotting(y1, label_x, label_y, freq, y1legend, *y):
    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    ax1.set_ylabel(label_y)
    ax1.set_xlabel(label_x)
    ax1.plot(pd.Series(np.arange(0, len(y1), 1) / freq * 1000), y1, label=y1legend[0])
    for count, value in enumerate(y):
        ax1.plot(pd.Series(np.arange(0, len(value), 1) / freq * 1000), value, label=y1legend[count+1])
    ax1.legend(loc='upper right')
    return fig

def interpolate_linear(y, length_new_y):
    x = np.linspace(0, len(y)-1, len(y))
    new_x = np.linspace(0, len(y) - 1, length_new_y)
    new_y = np.interp(new_x, x, y.squeeze())[np.newaxis].T
    return new_y

def interpolate(y, length_new_y): # Interpolate with cubic spline
    x = np.linspace(0, len(y) - 1, len(y))
    new_x = np.linspace(0, len(y) - 1, length_new_y)
    cs = sp.interpolate.CubicSpline(x, y.squeeze())
    new_y = cs(new_x)[np.newaxis].T  
    return new_y

def filt_lowpass(y, cutoff, frequency):
    #nyquist      = frequency * 0.5                            # Not necessary as fs is specified
    #norm_cutoff       = cutoff / nyquist              # normalize frequency
    b, a    = sp.signal.butter(2, cutoff, 'low', fs=frequency)#sp.signal.butter(2, w, 'low')
    new_y   = pd.DataFrame(sp.signal.filtfilt(b, a, y))
    return new_y

# def filt_lowpass(y, cutoff, frequency): # Savitzky-Golay filter
#     #nyquist      = frequency * 0.5                            # Not necessary as fs is specified
#     #norm_cutoff       = cutoff / nyquist              # normalize frequency
#     #b, a    = sp.signal.butter(2, cutoff, 'low', fs=frequency)#sp.signal.butter(2, w, 'low')
#     new_y   = pd.DataFrame(sp.signal.savgol_filter(y, window_length=7, polyorder=4))
#     return new_y

def filter_frequency(signal, frequency, percentile):
    # Frequency spectrum of signal
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    # Bins of FFT result 
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/frequency)  
    positive_frequencies = frequencies[:n // 2]         
    positive_magnitude = fft_magnitude[:n // 2] # only positive frequencies
    # Power spectrum & area under curve
    power_spectrum = positive_magnitude ** 2
    cumulative_power = np.cumsum(power_spectrum)
    cumulative_power /= cumulative_power[-1] # normalize
    # Find percentile
    index_percentile = np.where(cumulative_power >= (percentile/100))[0][0]
    cutoff_frequency = positive_frequencies[index_percentile]
    return round(cutoff_frequency)

def angle(marker_low_x, marker_low_y, 
          marker_mid_x, marker_mid_y, 
          marker_high_x, marker_high_y):
    slope_low       = []
    slope_upper     = []
    angle           = []
    """Slope lower segment"""
    for count, value in enumerate(marker_low_y):
        placeholder = ((value - marker_mid_y[count]) / 
                            (marker_low_x[count] - marker_mid_x[count]))
        slope_low.append(placeholder)
    """Slope upper segment"""
    for count, value in enumerate(marker_mid_y):
        if  (marker_mid_x[count] - marker_high_x[count]) == 0:
            placeholder = ((value - marker_high_y[count]) / 
                                (marker_mid_x[count]+0.00000001 - marker_high_x[count]))
        else:
            placeholder = ((value - marker_high_y[count]) / 
                            (marker_mid_x[count] - marker_high_x[count]))
        slope_upper.append(placeholder)
    """Angle"""
    for count, value in enumerate(slope_low):
        placeholder = np.rad2deg(np.arctan((slope_upper[count] - slope_low[count]) / 
                                                (1 + (slope_low[count] * slope_upper[count]))))
        if placeholder > 0:
            angle.append(placeholder)
        else:    
            angle.append(90 - (placeholder*-1) + 90) 
    return angle

# def angle(marker_low_x, marker_low_y, 
#           marker_mid_x, marker_mid_y, 
#           marker_high_x, marker_high_y):
#     # Convert inputs to numpy arrays for vectorized operations
#     marker_low_x, marker_low_y = np.array(marker_low_x), np.array(marker_low_y)
#     marker_mid_x, marker_mid_y = np.array(marker_mid_x), np.array(marker_mid_y)
#     marker_high_x, marker_high_y = np.array(marker_high_x), np.array(marker_high_y)   
#     # Calculate slopes for the lower and upper segments
#     slope_low = (marker_low_y - marker_mid_y) / (marker_low_x - marker_mid_x)
#     slope_upper = (marker_mid_y - marker_high_y) / (marker_mid_x - marker_high_x)    
#     # Calculate the angle between the two segments using the arctangent of the difference of slopes
#     angle_rad = np.arctan(np.abs((slope_upper - slope_low) / (1 + slope_low * slope_upper)))    
#     # Convert from radians to degrees
#     angle_deg = np.rad2deg(angle_rad)   
#     # Ensure the angle is positive (e.g., if negative angles are present, adjust them)
#     angle_deg = np.where(angle_deg < 0, 180 + angle_deg, angle_deg)    
#     return angle_deg


def acceleration(position_X, position_Y, freq):
    position_X = np.array(position_X)
    position_Y = np.array(position_Y)
    dt = 1.0 / freq
    velocity_X = np.diff(position_X) / dt
    velocity_Y = np.diff(position_Y) / dt
    acceleration_X = np.diff(velocity_X) / dt
    acceleration_Y = np.diff(velocity_Y) / dt
    acceleration_X = np.pad(acceleration_X, (2, 0), 'constant', constant_values=0)
    acceleration_Y = np.pad(acceleration_Y, (2, 0), 'constant', constant_values=0)
    return pd.DataFrame({
        'X': acceleration_X,
        'Y': acceleration_Y})

def angular_velocity(angle, freq):
    velocity     = []
    """ velocity """
    for count, value in enumerate(angle):
        if count < (len(angle) -1):
            placeholder = (angle[count + 1] - angle[count]) / (1 / freq)
            velocity.append(placeholder)                      
    velocity.insert(0, 0.0)
    velocity = np.array(velocity)
    return pd.DataFrame(velocity)

def angular_acceleration(angle, freq):
    angle = np.array(angle)
    dt = 1 / freq
    velocity = np.diff(angle) / dt
    acceleration = np.diff(velocity) / dt
    acceleration = np.pad(acceleration, (2, 0), 'constant', constant_values=0)
    return pd.DataFrame({'Angular_Acceleration': acceleration})

def reaction_forces(subj_mass, rel_mass_segm, segment_acc, force):
    segment_mass = subj_mass * rel_mass_segm * 2
    segment_reaction_force = pd.DataFrame()
    segment_reaction_force['X'] = (segment_mass * segment_acc['X']) - force['X']
    segment_reaction_force['Y'] = (segment_mass * segment_acc['Y']) + (segment_mass * 9.81) - force['Y']
    return pd.DataFrame(segment_reaction_force)

def proximal_moment(inertial_moment, segm_angle_acc, Moment_dist, 
                    distal_forces, proximal_forces, moment_arms):
    inertia = inertial_moment * segm_angle_acc
    prox_moment_vert  = proximal_forces['Y'] * moment_arms['dxp']
    prox_moment_horiz = proximal_forces['X'] * moment_arms['dyp']
    dist_moment_vert  = distal_forces['Y'] * moment_arms['dxd']
    dist_moment_horiz = distal_forces['X'] * moment_arms['dyd']
    Moment_proximal   = inertia - Moment_dist - (dist_moment_vert + dist_moment_horiz) - (prox_moment_vert + prox_moment_horiz)
    return Moment_proximal

def slope(x1, y1, x2, y2): # for residual analysis
  s = (y2-y1)/(x2-x1)
  return s