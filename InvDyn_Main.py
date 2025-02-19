# -*- coding: utf-8 -*-
"""
Inverse Dynamics for 2m Drop Landings

Created on Sat Apr  1 18:04:14 2023

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants
import json 
import os
import pickle
from functions_InvDyn import *

### Subject ###
subj           = 'Subject_08'
### Trial ###
trial          = 'Stiff'
### File ###
folder_subj    = 'C:\Daten\Studium RUB\Parkour Studie\InvDyn\data\\' + subj
folder_dir     = folder_subj + '/' + trial
file_dir       = 'C:\Daten\Studium RUB\Parkour Studie\InvDyn\data\\' + subj + '/' + trial + '//'
file_kin       = 'CIMG0001.json'
file_force     = 'CIMG0001_force.txt'
save_path      = 'C:\Daten\Studium RUB\Parkour Studie\InvDyn\output/'

"""Load Subject Scaling File"""
scaling = pd.read_table(folder_subj+'/'+subj+'_Scaling.txt', delimiter='\t', decimal='.', skiprows=3, header=None,
                           low_memory=False, index_col=False, names=['Variable', 'Value'])
    
for index, row in scaling.iterrows():
    globals()[row['Variable']] = pd.to_numeric(row['Value'])
    
### Measurment Specifics ###
freq_kin       = 300
freq_force     = 1200
# Relative segement mass in % of total mass
rel_mass_foot  = 0.0137
rel_mass_shank = 0.0433
rel_mass_thigh = 0.1478
# Longitudinal CoM of body segments from proximal/cranial end points in %
CM_foot        = 0.4415
CM_shank       = 0.4459
CM_thigh       = 0.4095
# Radius of Gyration (r) in % of segment length
gyration_foot  = 0.257
gyration_shank = 0.255
gyration_thigh = 0.329
# Earth acceleration
g              = sp.constants.g # 9.80665
# Lowpass cutoff-filter; Determined by residual analysis of 
# joint angles from 4 random subjects for each technique
cutoff_freq = {}
cutoff_freq['ankle']   = 43.0
cutoff_freq['knee']    = 37.0
cutoff_freq['hip']     = 28.0
# Start and End of final data to be saved
begin_ms        = 50 # 100ms before GC
if trial        == 'Stiff':
    stop_ms     = 350 # 100ms after peak knee flexion 
elif trial      == 'Normal':
    stop_ms     = 400 # 100ms after peak knee flexion
else:
    stop_ms     = 450 # 100ms after peak knee flexion 

""" Load in all files paths for one landing technique """
JSON_files  = []
FORCE_files = []
 
for root, dirs, files in os.walk(folder_dir):
    for file in files: 
 
        # change the extension from '.mp3' to 
        # the one of your choice.
        if file.endswith('.json'):
           JSON_files.append(root+'/'+str(file))
        elif file.endswith('.txt'):
           FORCE_files.append(root+'/'+str(file))
           
del file, files, root, dirs

variable_list = ['sole_X', 'sole_Y', 'metatarsiV_X',
                 'metatarsiV_Y', 'mall_lat_X', 'mall_lat_Y',
                 'epic_lat_X', 'epic_lat_Y', 'troch_maj_X',
                 'troch_maj_Y', 'tuberc_ilia_X', 'tuberc_ilia_Y',
                 'ankle_angle', 'knee_angle', 'hip_angle',
                 'ankle_vel', 'knee_vel', 'hip_vel',
                 'ankle_moment', 'knee_moment', 'hip_moment',
                 'ankle_compression', 'knee_compression', 'hip_compression',
                 'ankle_power', 'knee_power', 'hip_power',
                 'ankle_work', 'knee_work', 'hip_work',
                 'force_Y', 'force_X', 'time']
continuous_data = {}
mean_continuous = {}
std_continuous  = {}
mean_discrete   = {}
std_discrete    = {}
for count, value in enumerate(variable_list):
    continuous_data[value] = pd.DataFrame()
    mean_continuous[value] = pd.DataFrame()
    std_continuous[value] = pd.DataFrame()

for tick, number in enumerate(JSON_files):
    file_dir = ""
    file_kin = number
    file_force = FORCE_files[tick]

    with open(file_dir+file_kin) as f:
        data_json = json.load(f)
    
    markers = []
    for count, value in enumerate(data_json['data']['timeseries']):
        markers.append(data_json['data']['timeseries'][count]['name'])
        
    data = {}
    marker_lengths = []
    marker_start = []
    for count, value in enumerate(markers):
        data[str(value)] = pd.DataFrame(data_json['data']['timeseries'][count]['data']['0'])
        data[str(value)].rename(columns={int(0): 'X', int(1): 'Y'}, inplace=True)
        data[str(value)]['time'] = pd.DataFrame(data_json['data']['timeseries'][count]['time'])
        marker_lengths.append(len(data[markers[count]].X))
        marker_start.append(data[markers[count]].time[0])
    
    """Load force data"""
    data_force = pd.read_table(file_dir+file_force, delimiter='\t', decimal='.', skiprows=19, header=None,
                               low_memory=False, index_col=False, names=['time', 'Fy', 'Fz', 'Ay'])
    
    """Preparing motion tracking data"""    
    marker_sync       = {}    #(1) synchronise marker
    marker_filt       = {}    #(2) smooth marker movement with 10Hz lowpass filter (Shishov, Elabd, Komisar et al., 2021)
    ### (1) ###
    start_kin_raw     = max(marker_start)
    diff_start        = int((max(marker_start) - min(marker_start)) / (10/freq_kin)) # 10/freq_kin as kinovea changes the 300Hz to 30Hz
    len_kin_raw       = min(marker_lengths) - diff_start
    for count, value in enumerate(markers):
        x = data[value][data[value].time == max(marker_start)].index[0]
        marker_sync[str(value)]                = data[value][x:x + len_kin_raw].reset_index(drop=True)
    
    """ Cut marker data if longer than force data """
    if len(marker_sync['mall_lat']['Y']) / freq_kin > len(data_force['Fz']) / freq_force:
        for count, value in enumerate(markers):
            marker_sync[str(value)] = marker_sync[str(value)][0 : int(len(data_force['Fz']) / (freq_force/freq_kin) - 100)]
    
    # """  Alternative for filters """
    kinetics = ['Fy', 'Fz', 'Ay']
    filter_freq_force = pd.DataFrame(data=None, index=kinetics, columns=[0])
    filter_freq = pd.DataFrame(data=None, index=[markers], columns=['X', 'Y'])
    freq = -75
    freq_f = -300
    
    filter_freq['X'] = freq
    filter_freq['Y'] = freq
    
    filter_freq_force[0]['Fy'] = freq_f
    filter_freq_force[0]['Fz'] = freq_f
    filter_freq_force[0]['Ay'] = freq
    
    # Calculating offset to correct for it during the next step
    offset_corr = y_offset * 100 - np.min(marker_sync['metatarsiV'].Y)
    ###(2) ###
    if freq > 0:
        for count, value in enumerate(markers):
            marker_sync[str(value)].Y              = marker_sync[str(value)].Y + offset_corr # offset correction here
            marker_filt[str(value)]                = pd.DataFrame()
            marker_filt[str(value)][str('X')]      = filt_lowpass(marker_sync[value].X, filter_freq['X'][value], freq_kin)
            marker_filt[str(value)][str('Y')]      = filt_lowpass(marker_sync[value].Y, filter_freq['Y'][value], freq_kin)
            marker_filt[str(value)][str('X')]      = marker_filt[str(value)][str('X')] / 100 # cm to m
            marker_filt[str(value)][str('Y')]      = marker_filt[str(value)][str('Y')] / 100 # cm to m
    else:
        for count, value in enumerate(markers):
            marker_sync[str(value)].Y              = marker_sync[str(value)].Y + offset_corr # offset correction here
            marker_filt[str(value)]                = pd.DataFrame()
            marker_filt[str(value)][str('X')]      = marker_sync[value].X
            marker_filt[str(value)][str('Y')]      = marker_sync[value].Y
            marker_filt[str(value)][str('X')]      = marker_filt[str(value)][str('X')] / 100 # cm to m
            marker_filt[str(value)][str('Y')]      = marker_filt[str(value)][str('Y')] / 100 # cm to m

    """" Interpolate all data to fit force data """
    marker_interp                              = {}
    for count, value in enumerate(markers):
        marker_interp[str(value)]              = pd.DataFrame()
        marker_interp[str(value)][str('X')]    = pd.DataFrame(interpolate(marker_filt[value].X, int((len(marker_filt[value].X)*(freq_force/freq_kin)))))
        marker_interp[str(value)][str('Y')]    = pd.DataFrame(interpolate(marker_filt[value].Y, int((len(marker_filt[value].Y)*(freq_force/freq_kin)))))
        marker_interp[str(value)][str('X')]    = np.array(marker_interp[str(value)][str('X')])
        marker_interp[str(value)][str('Y')]    = np.array(marker_interp[str(value)][str('Y')])

    """Calculating joint angle"""
    joints                                     = ['ankle', 'knee', 'hip']
    joint_angle                                = {}
    joint_angle[joints[0]]                     = np.array(angle(marker_interp['metatarsiV'].X, marker_interp['metatarsiV'].Y, 
                                                                marker_interp['mall_lat'].X, marker_interp['mall_lat'].Y, 
                                                                marker_interp['epic_lat'].X, marker_interp['epic_lat'].Y))
    joint_angle[joints[0]]                     = (joint_angle[joints[0]] - ankle_offset) * -1# define quiet stance ankle angle as 0° ([-] = plantarflex.)
    joint_angle[joints[1]]                     = np.array(angle(marker_interp['mall_lat'].X, marker_interp['mall_lat'].Y, 
                                                                marker_interp['epic_lat'].X, marker_interp['epic_lat'].Y,
                                                                marker_interp['troch_maj'].X, marker_interp['troch_maj'].Y))
    joint_angle[joints[1]]                     = (joint_angle[joints[1]] + (180 - knee_offset)) * -1 # define quiet stance knee angle as 0° ([+] = knee ext.)
    joint_angle[joints[2]]                     = np.array(angle(marker_interp['epic_lat'].X, marker_interp['epic_lat'].Y,
                                                                marker_interp['troch_maj'].X, marker_interp['troch_maj'].Y,
                                                                marker_interp['tuberc_ilia'].X, marker_interp['tuberc_ilia'].Y))
    joint_angle[joints[2]]                     = (joint_angle[joints[2]] - hip_offset) # define quiet stance hip angle as 0° ([+] = hip ext.)

    """ Calculate joint angular velocity """
    joint_vel                                  = {}
    joint_vel[joints[0]]                       = angular_velocity((joint_angle['ankle'] * (np.pi / 180.0)), freq_force) # calculates °/s, convert to rad/s
    joint_vel[joints[1]]                       = angular_velocity((joint_angle['knee'] * (np.pi / 180.0)), freq_force) # calculates °/s, convert to rad/s
    joint_vel[joints[2]]                       = angular_velocity((joint_angle['hip'] * (np.pi / 180.0)), freq_force) # calculates °/s, convert to rad/s

    """ Calculate segemntal angular acceleration relative to horizontal """
    segments                                   = ['foot', 'shank', 'thigh'] # define body segments
    # Angular position of segment relative to horizonal (adding 0.5m to X of distal segmental end for a horizontal line)
    segment_angle                              = {}
    segment_angle[segments[0]]                 = np.array(angle(marker_interp['metatarsiV'].X + 0.5, marker_interp['metatarsiV'].Y,
                                                                marker_interp['metatarsiV'].X, marker_interp['metatarsiV'].Y,
                                                                marker_interp['mall_lat'].X, marker_interp['mall_lat'].Y))
    segment_angle[segments[1]]                 = np.array(angle(marker_interp['mall_lat'].X + 0.5, marker_interp['mall_lat'].Y,
                                                                marker_interp['mall_lat'].X, marker_interp['mall_lat'].Y,
                                                                marker_interp['epic_lat'].X, marker_interp['epic_lat'].Y))
    segment_angle[segments[2]]                 = np.array(angle(marker_interp['epic_lat'].X + 0.5, marker_interp['epic_lat'].Y,
                                                                marker_interp['epic_lat'].X, marker_interp['epic_lat'].Y,
                                                                marker_interp['troch_maj'].X, marker_interp['troch_maj'].Y))
    # Segmental angular acceleration in rad/s^2
    segment_angle_acc                          = {}
    segment_angle_acc[segments[0]]             = np.array(angular_acceleration(segment_angle['foot'], freq_force) * (np.pi / 180.0))  # from °/s^2 to rad/s^2
    segment_angle_acc[segments[1]]             = np.array(angular_acceleration(segment_angle['shank'], freq_force) * (np.pi / 180.0)) # from °/s^2 to rad/s^2
    segment_angle_acc[segments[2]]             = np.array(angular_acceleration(segment_angle['thigh'], freq_force) * (np.pi / 180.0)) # from °/s^2 to rad/s^2

    """ Calculate linear acceleration of CoM in X and Y direction """
    # CoM in X and Y direction per body segment
    segment_CoM                                = {}
    segment_CoM[segments[0]]                   = marker_interp['mall_lat'] - (CM_foot * (marker_interp['mall_lat'] - marker_interp['metatarsiV']))
    segment_CoM[segments[1]]                   = marker_interp['epic_lat'] - (CM_shank * (marker_interp['epic_lat'] - marker_interp['mall_lat']))
    segment_CoM[segments[2]]                   = marker_interp['troch_maj'] - (CM_thigh * (marker_interp['troch_maj'] - marker_interp['epic_lat']))
    # Acceleration in X and Y direction per body segement based on filtered non-interpolated data
    segment_acc                                = {}
    segment_acc[segments[0]]                   = acceleration(segment_CoM['foot']['X'], segment_CoM['foot']['Y'], freq_force)
    segment_acc[segments[1]]                   = acceleration(segment_CoM['shank']['X'], segment_CoM['shank']['Y'], freq_force)
    segment_acc[segments[2]]                   = acceleration(segment_CoM['thigh']['X'], segment_CoM['thigh']['Y'], freq_force)

    """ Synchronize kinematic to force data """
    # Calculate velocity of lowest marker (metatarsal) from raw data to sync. kinetic & kinemtaic data 
    metatarsiV_vel                             = [] 
    for count, value in enumerate(marker_sync['metatarsiV'].Y):
        if count < (len(marker_sync['metatarsiV'].Y) - 1):
            velocity                           = (marker_sync['metatarsiV'].Y[count + 1]/100 - value/100) / (1 / freq_kin)
            metatarsiV_vel.append(velocity)
        else:
            del velocity
    metatarsiV_vel                             = np.array(metatarsiV_vel)
    # Interpolate
    metatarsiV_vel_interp                      = interpolate_linear(metatarsiV_vel, len(marker_interp['metatarsiV'].Y))
    # Define points for synchronisation right at impact
    GC_kin                                     = (next(x for x, val in enumerate(metatarsiV_vel_interp[100:]) if val > -3.0)) -4 + 100 # impact just after max velocity (3.0m/s^2 - 4 frames = ~ 4m/s^2)
    GC_force                                   = next(x for x, val in enumerate(data_force['Fz']) if val > 50)  # impact at vertical force above 50N
    # Synchronise force to kinematic data

    ### Filter force data ###
    if freq > 0:
        force                                      = pd.DataFrame()
        force['Fz']                                = filt_lowpass(data_force['Fz'], filter_freq_force[0]['Fz'], freq_force)
        force['Fy']                                = filt_lowpass(data_force['Fy'], filter_freq_force[0]['Fy'], freq_force)
        force['Ay']                                = data_force['Ay']
        force['Ay'][0:GC_force]                    = 0.00
        force['Ay']                                = filt_lowpass(force['Ay'], filter_freq_force[0]['Ay'], freq_force)
    else:
        force                                      = pd.DataFrame()
        force['Fz']                                = data_force['Fz']
        force['Fy']                                = data_force['Fy']
        force['Ay']                                = data_force['Ay']
        force['Ay'][0:GC_force]                    = 0.00

    force_sync                                     = {}
    if (GC_force - GC_kin) > 0:
        for count, value in enumerate(force):
            force_sync[str(value)]                 = np.array(force[str(value)][GC_force - GC_kin : 
                                                            GC_force + len(marker_interp['metatarsiV']) - GC_kin].reset_index(drop=True))
        force_sync['Y']                            = force_sync.pop('Fz') # vertical forces
        force_sync['X']                            = force_sync.pop('Fy') # horizontal forces
        #force_sync['Y'][0:GC_kin]                  = 0.00        # All data on force plates before GC should be zero
        #force_sync['X'][0:GC_kin]                  = 0.00        # All data on force plates before GC should be zero
        #force_sync['Ay'][0:GC_kin+3]                 = 0.00        # All data on force plates before GC should be zero   
    else:
        for count, value in enumerate(force):
            num_zeros                              = GC_kin
            pad_with_zeros                         = np.zeros(num_zeros)
            force_sync[str(value)]                 = np.array(force[str(value)][GC_force : 
                                                           GC_force + len(marker_interp['metatarsiV']) - GC_kin].reset_index(drop=True))
            force_sync[str(value)]                 = np.insert(force_sync[str(value)], 0, pad_with_zeros)
        force_sync['Y']                        = force_sync.pop('Fz') # vertical forces
        force_sync['X']                        = force_sync.pop('Fy') # horizontal forces
     
    """ Define CoP """                                       # Initial contact with metatarsii is expected, thus calculations from there
    CoP_offset = force_sync['Ay'][GC_kin] + IC_distance
    CoP                                        = marker_interp['metatarsiV'].X - force_sync['Ay'] + CoP_offset # CoP in ground thus only horizontal component
    CoP[0:GC_kin]                              = 0.00        # No CoP data before GC, thus equal zero

    """ Calculate Moment of Inertia for all segments """
    inertial_moment                            = {}
    inertial_moment[segments[0]]               = (subj_mass * rel_mass_foot * 2) * (np.square((length_foot * gyration_foot)))
    inertial_moment[segments[1]]               = (subj_mass * rel_mass_shank * 2) * (np.square((length_shank * gyration_shank)))
    inertial_moment[segments[2]]               = (subj_mass * rel_mass_thigh * 2) * (np.square((length_thigh * gyration_thigh)))

    """ Calculate Joint Moments """
    ### Joint Reaction Forces ###
    reaction = {}
    reaction['ankle']                          = reaction_forces(subj_mass, rel_mass_foot, segment_acc['foot'], force_sync)
    reaction['knee']                           = reaction_forces(subj_mass, rel_mass_shank, segment_acc['shank'], (reaction['ankle'] * -1))
    reaction['hip']                            = reaction_forces(subj_mass, rel_mass_thigh, segment_acc['thigh'], (reaction['knee'] * -1))

    reaction_filt = {}
    for count, value in enumerate(joints):
        reaction_filt[value] = pd.DataFrame()
        reaction_filt[value]['X'] = filt_lowpass(reaction[value]['X'], cutoff_freq[value], freq_force)
        reaction_filt[value]['Y'] = filt_lowpass(reaction[value]['Y'], cutoff_freq[value], freq_force)

    ### Joint Moment Arms ###
    ankle_moment_arms                          = {}
    ankle_moment_arms['dxp']                   = marker_interp['mall_lat'].X - segment_CoM['foot'].X # joint_prox_x - CoM_x  !NEGATIVE!
    ankle_moment_arms['dyp']                   = segment_CoM['foot'].Y - marker_interp['mall_lat'].Y # CoM_y - joint_prox_y 
    ankle_moment_arms['dxd']                   = CoP - segment_CoM['foot'].X                             # joint_dist_x - CoM_x     
    ankle_moment_arms['dyd']                   = segment_CoM['foot'].Y - 0.00                            # CoM_y - joint_dist_y !POSITIVE! CoP in ground

    knee_moment_arms                           = {}
    knee_moment_arms['dxp']                    = marker_interp['epic_lat'].X - segment_CoM['shank'].X 
    knee_moment_arms['dyp']                    = segment_CoM['shank'].Y - marker_interp['epic_lat'].Y 
    knee_moment_arms['dxd']                    = marker_interp['mall_lat'].X - segment_CoM['shank'].X                               
    knee_moment_arms['dyd']                    = segment_CoM['shank'].Y - marker_interp['mall_lat'].Y 

    hip_moment_arms                            = {}
    hip_moment_arms['dxp']                     = marker_interp['troch_maj'].X - segment_CoM['thigh'].X 
    hip_moment_arms['dyp']                     = segment_CoM['thigh'].Y - marker_interp['troch_maj'].Y 
    hip_moment_arms['dxd']                     = marker_interp['epic_lat'].X - segment_CoM['thigh'].X                               
    hip_moment_arms['dyd']                     = segment_CoM['thigh'].Y - marker_interp['epic_lat'].Y                           

    ### Joint Moments ### Equation: M_prox = Inertia - M_dist - ((Ryd * dxd) + (Rxd * dyd)) - ((Ryp * dxp) + (Rxp * dyp))
    ankle_moment                               = proximal_moment(inertial_moment['foot'], segment_angle_acc['foot'][:,0], 
                                                                 0.00, force_sync, reaction['ankle'], ankle_moment_arms)

    knee_moment                                = proximal_moment(inertial_moment['shank'], segment_angle_acc['shank'][:,0], 
                                                                 (ankle_moment * -1), (reaction['ankle'] * -1), reaction['knee'], knee_moment_arms)

    hip_moment                                 = proximal_moment(inertial_moment['thigh'], segment_angle_acc['thigh'][:,0], 
                                                                 (knee_moment * -1), (reaction['knee'] * -1), reaction['hip'], hip_moment_arms) * -1

    moments_filt = {}
    moments_filt[joints[0]] = filt_lowpass(ankle_moment, cutoff_freq['ankle'], freq_force)
    moments_filt[joints[1]] = filt_lowpass(knee_moment, cutoff_freq['knee'], freq_force)
    moments_filt[joints[2]] = filt_lowpass(hip_moment, cutoff_freq['hip'], freq_force)

    """ Calculate Joint Power """
    ### Joint Moment * Joint Velocity ###
    ankle_power                                = moments_filt['ankle'] * filt_lowpass(joint_vel['ankle'][0], cutoff_freq['ankle'], freq_force)
    knee_power                                 = moments_filt['knee'] * filt_lowpass(joint_vel['knee'][0], cutoff_freq['knee'], freq_force)
    hip_power                                  = moments_filt['hip'] * filt_lowpass(joint_vel['hip'][0], cutoff_freq['hip'], freq_force)
    
    """ Calculate Joint Work """
    ankle_work                                 = np.cumsum((ankle_power[:-1] + ankle_power[1:]) / 2) * (1/freq_force)
    knee_work                                  = np.cumsum((knee_power[:-1] + knee_power[1:]) / 2) * (1/freq_force)
    hip_work                                   = np.cumsum((hip_power[:-1] + hip_power[1:]) / 2) * (1/freq_force)
    
    """ Calculate axial loading (compression) of joints """
    # For hip and knee joint the body segments can be taken, for the ankle a line
    # between the malleolus and a point underneath the foot (90° to sole of shoe)
    # is calculated:
        # (1) Angle of mall lat from horizontal when foot flat (arch) 
    min_mall_lat = np.argmin(marker_interp['mall_lat']['Y'])
    arch_foot_angle = 180 - segment_angle['foot'][min_mall_lat]
    angle_sole = 180 - segment_angle['foot'] - arch_foot_angle
        # (2) Distance of point underneath foot by adjacent * cos or sin of angle of soel of foot
    arch_hypothenuse = np.sqrt((marker_interp['mall_lat']['X'] - marker_interp['metatarsiV']['X'])**2 +
                               (marker_interp['mall_lat']['Y'] - marker_interp['metatarsiV']['Y'])**2)
    arch_adjacent = np.cos(np.deg2rad(arch_foot_angle)) * arch_hypothenuse
    arch_opposite = np.sin(np.deg2rad(arch_foot_angle)) * arch_hypothenuse
    
    foot_sole = pd.DataFrame()
    foot_sole['X'] = marker_interp['metatarsiV']['X'] - (arch_adjacent * np.cos(np.deg2rad(angle_sole)))
    foot_sole['Y'] = marker_interp['mall_lat']['Y'] - (arch_opposite * np.cos(np.deg2rad(angle_sole)))
    
    angle_ankle_comp = np.array(angle(foot_sole.X + 0.5, foot_sole.Y, foot_sole.X, foot_sole.Y, marker_interp['mall_lat'].X, marker_interp['mall_lat'].Y))

    # Calculate compression force for hip & knee
    resultant   = {}
    compression = {}
    for count, value in enumerate(joints):
        resultant[value] = pd.DataFrame()
        # Calculate resultant force from X & Y component [Fres = sqrt(Fx^2 + Fy^2)]
        resultant[value]['force']              = np.sqrt(reaction_filt[value]['X']**2 + reaction_filt[value]['Y']**2)
        # Calculate angle of resultant force [AngleRes = arctan(Fy / Fx)]
        resultant_angle                        = np.rad2deg(np.arctan2(reaction_filt[value]['Y'], reaction_filt[value]['X']))
        # Change force angle to fit segment angle coordinate system
        resultant[value]['angle']              = np.where(resultant_angle < 0, -resultant_angle, -resultant_angle + 180)
        # Calculate axial force of joints [Fcomp = Fres * cos(SegmentAngle - AngeRes)]
        compression[value]                     = resultant[value]['force'] * np.cos(np.deg2rad(segment_angle[segments[count]] - resultant[value]['angle']))  
     
    # Re-calculate compression force for ankle
    compression['ankle']                     = resultant['ankle']['force'] * np.cos(np.deg2rad(angle_ankle_comp - resultant['ankle']['angle']))

    """ Find Timing of start, end & impact phase """
    start                                      = int(GC_kin - (freq_force / 1000 * begin_ms)) # start 50ms before GC
    end                                        = int(GC_kin + (freq_force / 1000 * stop_ms)) # end 350ms & 450 ms after GC for stiff, soft and normal, respectivly
    
    """ Append all Trials of One Landing Technqiue to one Dictionary """  
    continuous_data['sole_X']                    = pd.concat([continuous_data['sole_X'], pd.DataFrame(foot_sole['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['sole_Y']                    = pd.concat([continuous_data['sole_Y'], pd.DataFrame(foot_sole['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['metatarsiV_X']              = pd.concat([continuous_data['metatarsiV_X'], pd.DataFrame(marker_interp['metatarsiV']['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['metatarsiV_Y']              = pd.concat([continuous_data['metatarsiV_Y'], pd.DataFrame(marker_interp['metatarsiV']['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['mall_lat_X']                = pd.concat([continuous_data['mall_lat_X'], pd.DataFrame(marker_interp['mall_lat']['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['mall_lat_Y']                = pd.concat([continuous_data['mall_lat_Y'], pd.DataFrame(marker_interp['mall_lat']['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['epic_lat_X']                = pd.concat([continuous_data['epic_lat_X'], pd.DataFrame(marker_interp['epic_lat']['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['epic_lat_Y']                = pd.concat([continuous_data['epic_lat_Y'], pd.DataFrame(marker_interp['epic_lat']['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['troch_maj_X']               = pd.concat([continuous_data['troch_maj_X'], pd.DataFrame(marker_interp['troch_maj']['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['troch_maj_Y']               = pd.concat([continuous_data['troch_maj_Y'], pd.DataFrame(marker_interp['troch_maj']['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['tuberc_ilia_X']             = pd.concat([continuous_data['tuberc_ilia_X'], pd.DataFrame(marker_interp['tuberc_ilia']['X'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['tuberc_ilia_Y']             = pd.concat([continuous_data['tuberc_ilia_Y'], pd.DataFrame(marker_interp['tuberc_ilia']['Y'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
 
    continuous_data['ankle_angle']               = pd.concat([continuous_data['ankle_angle'], pd.DataFrame(joint_angle['ankle'][start:end] * -1).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['knee_angle']                = pd.concat([continuous_data['knee_angle'], pd.DataFrame(joint_angle['knee'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['hip_angle']                 = pd.concat([continuous_data['hip_angle'], pd.DataFrame(joint_angle['hip'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)

    continuous_data['ankle_vel']                 = pd.concat([continuous_data['ankle_vel'], pd.DataFrame(joint_vel['ankle'][start:end] * -1).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['knee_vel']                  = pd.concat([continuous_data['knee_vel'], pd.DataFrame(joint_vel['knee'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['hip_vel']                   = pd.concat([continuous_data['hip_vel'], pd.DataFrame(joint_vel['hip'][start:end]).reset_index(drop=True)], ignore_index=True, axis=1)

    continuous_data['ankle_moment']              = pd.concat([continuous_data['ankle_moment'], pd.DataFrame((moments_filt['ankle'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['knee_moment']               = pd.concat([continuous_data['knee_moment'], pd.DataFrame((moments_filt['knee'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['hip_moment']                = pd.concat([continuous_data['hip_moment'], pd.DataFrame((moments_filt['hip'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)

    continuous_data['ankle_compression']         = pd.concat([continuous_data['ankle_compression'], pd.DataFrame((compression['ankle'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['knee_compression']          = pd.concat([continuous_data['knee_compression'], pd.DataFrame((compression['knee'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)                                                       
    continuous_data['hip_compression']           = pd.concat([continuous_data['hip_compression'], pd.DataFrame((compression['hip'][start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)                                                         
                                                             
    continuous_data['ankle_power']               = pd.concat([continuous_data['ankle_power'], pd.DataFrame((ankle_power[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['knee_power']                = pd.concat([continuous_data['knee_power'], pd.DataFrame((knee_power[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['hip_power']                 = pd.concat([continuous_data['hip_power'], pd.DataFrame((hip_power[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)

    continuous_data['ankle_work']                = pd.concat([continuous_data['ankle_work'], pd.DataFrame((ankle_work[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['knee_work']                 = pd.concat([continuous_data['knee_work'], pd.DataFrame((knee_work[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)
    continuous_data['hip_work']                  = pd.concat([continuous_data['hip_work'], pd.DataFrame((hip_work[start:end] / subj_mass).reset_index(drop=True))], ignore_index=True, axis=1)

    continuous_data['force_Y']                   = pd.concat([continuous_data['force_Y'], pd.DataFrame(force_sync['Y'][start:end] / subj_mass).reset_index(drop=True)], ignore_index=True, axis=1)
    continuous_data['force_X']                   = pd.concat([continuous_data['force_X'], pd.DataFrame(force_sync['X'][start:end] / subj_mass).reset_index(drop=True)], ignore_index=True, axis=1)

    continuous_data['time']                      = pd.concat([continuous_data['time'] , pd.DataFrame(np.arange(begin_ms * -1, stop_ms, 1000/freq_force))], ignore_index=True, axis=1)

for count, value in enumerate(joints):
    continuous_data[value + '_angle'].columns    = [f'trial_{i+1}' for i in range(len(JSON_files))]
    continuous_data[value + '_vel'].columns      = [f'trial_{i+1}' for i in range(len(JSON_files))]
    continuous_data[value + '_moment'].columns   = [f'trial_{i+1}' for i in range(len(JSON_files))]
    continuous_data[value + '_power'].columns    = [f'trial_{i+1}' for i in range(len(JSON_files))]
    continuous_data[value + '_work'].columns     = [f'trial_{i+1}' for i in range(len(JSON_files))]
    continuous_data[value + '_compression'].columns = [f'trial_{i+1}' for i in range(len(JSON_files))]    
continuous_data['force_Y'].columns               = [f'trial_{i+1}' for i in range(len(JSON_files))]
continuous_data['force_X'].columns               = [f'trial_{i+1}' for i in range(len(JSON_files))]
continuous_data['time'].columns                  = [f'trial_{i+1}' for i in range(len(JSON_files))]

""" Calculate discrete Values """
discrete_values = pd.DataFrame()
for tick, number in enumerate(JSON_files):
    # Find start impact
    GC                  = int(begin_ms / 1000 * freq_force)
    # Find end impact
    force_Y_data        = continuous_data['force_Y']['trial_' + str(tick + 1)]
    peaks, _            = sp.signal.find_peaks(force_Y_data, distance=20)           # find all peaks
    threshold           = 0.50 * np.max(force_Y_data)                  # only peaks above 60% of force
    high_peaks          = peaks[force_Y_data[peaks] > threshold]     # 2 peaks
    peak_1_time         = high_peaks[0]
    peak_1              = force_Y_data[peak_1_time]
    peak_2_time         = high_peaks[1]
    peak_2              = force_Y_data[peak_2_time]
    troughs, _          = sp.signal.find_peaks(-force_Y_data)        # Invert data to find troughs
    troughs_after_peak  = troughs[troughs > peak_2_time]
    # End of impact only counts if value above 55ms after GC
    end_impact_index    = int(next(x for x, val in enumerate(troughs_after_peak) if ((val - GC) / freq_force * 1000) > 50))
    end_impact          = int(troughs_after_peak[end_impact_index])   
      
    
    discrete_values.at['1st_Peak', 'trial_' + str(tick + 1)]                        =  peak_1
    discrete_values.at['2nd_Peak', 'trial_' + str(tick + 1)]                        =  peak_2
    discrete_values.at['1st_Peak_time', 'trial_' + str(tick + 1)]                   =  (peak_1_time - GC) / freq_force * 1000
    discrete_values.at['2nd_Peak_time', 'trial_' + str(tick + 1)]                   =  (peak_2_time - GC) / freq_force * 1000
    discrete_values.at['impact_length', 'trial_' + str(tick + 1)]                   =  (end_impact - GC)  / freq_force * 1000

    
    
    # Split power into concentric (positive) and eccentric (negative)
    positive_power = pd.DataFrame()
    negative_power = pd.DataFrame()
    positive_moment = pd.DataFrame()
    negative_moment = pd.DataFrame()
    for count, value in enumerate(joints):
        positive_power[value] = np.where(continuous_data[value + '_power']['trial_' + str(tick + 1)][GC:end_impact] > 0,
                                         continuous_data[value +'_power']['trial_' + str(tick + 1)][GC:end_impact],0)
        negative_power[value] = np.where(continuous_data[value +'_power']['trial_' + str(tick + 1)][GC:end_impact] < 0,
                                         continuous_data[value +'_power']['trial_' + str(tick + 1)][GC:end_impact],0)
        
        positive_moment[value] = np.where(continuous_data[value + '_moment']['trial_' + str(tick + 1)][GC:end_impact] > 0,
                                         continuous_data[value +'_moment']['trial_' + str(tick + 1)][GC:end_impact],0)
        negative_moment[value] = np.where(continuous_data[value +'_moment']['trial_' + str(tick + 1)][GC:end_impact] < 0,
                                         continuous_data[value +'_moment']['trial_' + str(tick + 1)][GC:end_impact],0)
        
        discrete_values.at[value +'_work_conc', 'trial_' + str(tick + 1)]             =  np.trapz(positive_power[value], dx=1/freq_force)
        discrete_values.at[value +'_work_ecc', 'trial_' + str(tick + 1)]             =  np.trapz(negative_power[value], dx=1/freq_force)
        discrete_values.at[value +'_impulse_ext', 'trial_' + str(tick + 1)]             =  np.trapz(positive_moment[value], dx=1/freq_force)
        discrete_values.at[value +'_impulse_flex', 'trial_' + str(tick + 1)]             =  np.trapz(negative_moment[value], dx=1/freq_force)

 
    
    # discrete_values.at['ankle_impulse', 'trial_' + str(tick + 1)]                   =  np.trapz(continuous_data['ankle_moment']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force)
    # discrete_values.at['knee_impulse', 'trial_' + str(tick + 1)]                    =  np.trapz(continuous_data['knee_moment']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force)
    # discrete_values.at['hip_impulse', 'trial_' + str(tick + 1)]                     =  np.trapz(continuous_data['hip_moment']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force)
    # discrete_values.at['ankle_work', 'trial_' + str(tick + 1)]                      =  np.trapz(continuous_data['ankle_power']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force)
    # discrete_values.at['knee_work', 'trial_' + str(tick + 1)]                       =  np.trapz(continuous_data['knee_power']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force)
    # discrete_values.at['hip_work', 'trial_' + str(tick + 1)]                        =  np.trapz(continuous_data['hip_power']['trial_' + str(tick + 1)][GC:end_impact], dx=1/freq_force) 
    discrete_values.at['P_ankle_comp', 'trial_' + str(tick + 1)]                    =  np.max(continuous_data['ankle_compression']['trial_' + str(tick + 1)])
    discrete_values.at['P_knee_comp', 'trial_' + str(tick + 1)]                     =  np.max(continuous_data['knee_compression']['trial_' + str(tick + 1)])
    discrete_values.at['P_hip_comp', 'trial_' + str(tick + 1)]                      =  np.max(continuous_data['hip_compression']['trial_' + str(tick + 1)][GC:GC + 200])
    
    discrete_values.at['P_ankle_moment', 'trial_' + str(tick + 1)]                  =  np.min(continuous_data['ankle_moment']['trial_' + str(tick + 1)][GC:end_impact])
    discrete_values.at['P_knee_moment', 'trial_' + str(tick + 1)]                   =  np.max(continuous_data['knee_moment']['trial_' + str(tick + 1)][GC:end_impact])
    discrete_values.at['P_hip_moment', 'trial_' + str(tick + 1)]                    =  np.max(continuous_data['hip_moment']['trial_' + str(tick + 1)][GC:end_impact])

    discrete_values.at['P_ankle_angle(impact)', 'trial_' + str(tick + 1)]           =  np.min(continuous_data['ankle_angle']['trial_' + str(tick + 1)][GC:end_impact])
    discrete_values.at['P_knee_angle', 'trial_' + str(tick + 1)]                    =  np.min(continuous_data['knee_angle']['trial_' + str(tick + 1)])
    discrete_values.at['P_hip_angle', 'trial_' + str(tick + 1)]                     =  np.min(continuous_data['hip_angle']['trial_' + str(tick + 1)])

    discrete_values.at['Ankle_GC', 'trial_' + str(tick + 1)]                        =  continuous_data['ankle_angle']['trial_' + str(tick + 1)][GC]
    discrete_values.at['Knee_GC', 'trial_' + str(tick + 1)]                         =  continuous_data['knee_angle']['trial_' + str(tick + 1)][GC]
    discrete_values.at['Hip_GC', 'trial_' + str(tick + 1)]                          =  continuous_data['hip_angle']['trial_' + str(tick + 1)][GC]

    discrete_values.at['P_ankle_Vel', 'trial_' + str(tick + 1)]                     =  np.min(continuous_data['ankle_vel']['trial_' + str(tick + 1)] * (180 / np.pi))
    discrete_values.at['P_knee_Vel', 'trial_' + str(tick + 1)]                      =  np.min(continuous_data['knee_vel']['trial_' + str(tick + 1)] * (180 / np.pi))
    discrete_values.at['P_hip_Vel', 'trial_' + str(tick + 1)]                       =  np.min(continuous_data['hip_vel']['trial_' + str(tick + 1)] * (180 / np.pi))
    
    discrete_values.at['P_ankle_power_ecc', 'trial_' + str(tick + 1)]               =  np.min(continuous_data['ankle_power']['trial_' + str(tick + 1)])
    discrete_values.at['P_knee_power_ecc', 'trial_' + str(tick + 1)]                =  np.min(continuous_data['knee_power']['trial_' + str(tick + 1)])
    discrete_values.at['P_hip_power_ecc', 'trial_' + str(tick + 1)]                 =  np.min(continuous_data['hip_power']['trial_' + str(tick + 1)])

    discrete_values.at['P_ankle_power_conc', 'trial_' + str(tick + 1)]              =  np.max(continuous_data['ankle_power']['trial_' + str(tick + 1)][0:GC + 200])
    discrete_values.at['P_knee_power_conc', 'trial_' + str(tick + 1)]               =  np.max(continuous_data['knee_power']['trial_' + str(tick + 1)][0:GC + 200])
    discrete_values.at['P_hip_power_conc', 'trial_' + str(tick + 1)]                =  np.max(continuous_data['hip_power']['trial_' + str(tick + 1)][0:GC + 150])

    discrete_values.at['t_P_ankle_moment', 'trial_' + str(tick + 1)]                =  np.argmin(continuous_data['ankle_moment']['trial_' + str(tick + 1)][GC:end_impact]) / freq_force * 1000
    discrete_values.at['t_P_knee_moment', 'trial_' + str(tick + 1)]                 =  np.argmax(continuous_data['knee_moment']['trial_' + str(tick + 1)][GC:end_impact]) / freq_force * 1000
    discrete_values.at['t_P_hip_moment', 'trial_' + str(tick + 1)]                  =  np.argmax(continuous_data['hip_moment']['trial_' + str(tick + 1)][GC:end_impact]) / freq_force * 1000

    discrete_values.at['t_P_ankle_angle(impact)', 'trial_' + str(tick + 1)]         =  np.argmin(continuous_data['ankle_angle']['trial_' + str(tick + 1)][GC:end_impact]) / freq_force * 1000
    discrete_values.at['t_P_knee_angle', 'trial_' + str(tick + 1)]                  =  (np.argmin(continuous_data['knee_angle']['trial_' + str(tick + 1)]) -GC) / freq_force * 1000
    discrete_values.at['t_P_hip_angle', 'trial_' + str(tick + 1)]                   =  (np.argmin(continuous_data['hip_angle']['trial_' + str(tick + 1)]) -GC) / freq_force * 1000

    discrete_values.at['t_P_ankle_Vel', 'trial_' + str(tick + 1)]                   =  (np.argmin(continuous_data['ankle_vel']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000
    discrete_values.at['t_P_knee_Vel', 'trial_' + str(tick + 1)]                    =  (np.argmin(continuous_data['knee_vel']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000
    discrete_values.at['t_P_hip_Vel', 'trial_' + str(tick + 1)]                     =  (np.argmin(continuous_data['hip_vel']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000


    discrete_values.at['t_P_ankle_power_ecc', 'trial_' + str(tick + 1)]             =  (np.argmin(continuous_data['ankle_power']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000
    discrete_values.at['t_P_knee_power_ecc', 'trial_' + str(tick + 1)]              =  (np.argmin(continuous_data['knee_power']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000
    discrete_values.at['t_P_hip_power_ecc', 'trial_' + str(tick + 1)]               =  (np.argmin(continuous_data['hip_power']['trial_' + str(tick + 1)]) - GC) / freq_force * 1000

    discrete_values.at['t_P_ankle_power_conc', 'trial_' + str(tick + 1)]            =  (np.argmax(continuous_data['ankle_power']['trial_' + str(tick + 1)][0:GC + 200]) - GC)  / freq_force * 1000
    discrete_values.at['t_P_knee_power_conc', 'trial_' + str(tick + 1)]             =  (np.argmax(continuous_data['knee_power']['trial_' + str(tick + 1)][0:GC + 200]) - GC) / freq_force * 1000
    discrete_values.at['t_P_hip_power_conc', 'trial_' + str(tick + 1)]              =  (np.argmax(continuous_data['hip_power']['trial_' + str(tick + 1)][0:GC + 150]) - GC) / freq_force * 1000

""" Mean Continuous Values of one Landing Technique """
for count, value in enumerate(variable_list):
    mean_continuous[value]  = np.mean(continuous_data[value], axis=1)
    std_continuous[value]   = np.std(continuous_data[value], axis=1)

""" Mean Discrete Values of one Landing Technique """    
mean_discrete               = np.mean(discrete_values, axis=1)
std_discrete                = np.std(discrete_values, axis=1)
 
""" Save Data to Output folder """
# Save gathered data (1), mean values (2), standard deviation (3)
# and save all also as python files [(4), (5), (6)]
### (1) ###
with open(save_path + subj + '_' + trial + '_continuous_data.txt', 'w') as f:
    for key, df in continuous_data.items():
        f.write(f'#{key} \n')
        f.write(df.to_csv(sep=';', index=False))
        f.write('\n')  # Add an extra newline to separate DataFrames
with open(save_path + subj + '_' + trial + '_discrete_values.txt', 'w') as f:
    # Write the column headers (the names of the Series in the DataFrame)
    f.write('Index ' + ' '.join([str(col) for col in discrete_values.columns]) + '\n') 
    # Loop through the DataFrame row by row and write each row
    for idx, row in discrete_values.iterrows():
        # Convert each row to a space-separated string
        row_string = f'{idx} ' + ' '.join([str(value) for value in row.values])
        f.write(row_string + '\n')  # Write the row string to the file

### (4) ###
with open(save_path + subj + '_' + trial + '_continuous_data.pkl', 'wb') as f:
    pickle.dump(continuous_data, f)
with open(save_path + subj + '_' + trial + '_discrete_values.pkl', 'wb') as f:
    pickle.dump(discrete_values, f)






# """ Save Mean Arrays (not necessary atm) """
# ### (2) ###       
# with open(save_path + subj + '_' + trial + '_meanvalues.txt', 'w') as f:
#     for key, df in mean_continuous.items():
#         f.write(f'#{key} \n')
#         f.write(df.to_csv(sep=';', index=False))
#         f.write('\n')  # Add an extra newline to separate DataFrames
# ### (3) ###        
# with open(save_path + subj + '_' + trial + '_stdvalues.txt', 'w') as f:
#     for key, df in std_continuous.items():
#         f.write(f'#{key} \n')
#         f.write(df.to_csv(sep=';', index=False))
#         f.write('\n')  # Add an extra newline to separate DataFrames
# ### (5) ###
# with open(save_path + subj + '_' + trial + '_mean_values.pkl', 'wb') as f:
#     pickle.dump(mean_continuous, f)
# ### (6) ###
# with open(save_path + subj + '_' + trial + '_std_values.pkl', 'wb') as f:
#     pickle.dump(std_continuous, f)


# """ Plots """
# begin_ms = 50
# stop_ms = 300
# start = int(GC_kin - (begin_ms * freq_force/1/1000))
# end = int(GC_kin + (stop_ms * freq_force/1/1000))
# time = (np.arange((stop_ms * freq_force/1/1000) - len(ankle_moment[start:end]), len(ankle_moment[start:end]) - (begin_ms * freq_force/1/1000), 1) / freq_force * 1000)
# ### Forces ###
# fig, (ax1) = plt.subplots(nrows=1, ncols=1)
# ax1.plot(time, force_sync['Y'][start:end]/subj_mass, color='black')
# ax1.plot(time, force_sync['X'][start:end]/subj_mass, color='blue')
# ax1.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# #ax2.plot(time, force_sync['Ay'][start:end]/subj_mass, color='black')
# #ax2.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ### Kinematics ###
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
# ax1.plot(time, (joint_angle_interp['ankle'][start:end])*-1, color='black')
# ax1.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax2.plot(time, joint_angle_interp['knee'][start:end], color='black')
# ax2.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax3.plot(time, joint_angle_interp['hip'][start:end], color='black')
# ax3.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax1.set_title('Joint Angles', fontsize=30)
# ax1.set_ylabel('Ankle [°]', fontsize=20)
# ax2.set_ylabel('Knee [°]', fontsize=20)
# ax3.set_ylabel('Hip [°]', fontsize=20)
# ax3.set_xlabel('Time [ms]', fontsize=20)
# ### Moments ###
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
# ax1.set_title('Joint Moments', fontsize=30)
# ax1.set_ylabel('Ankle [Nm/kg]', fontsize=20)
# ax2.set_ylabel('Knee [Nm/kg]', fontsize=20)
# ax3.set_ylabel('Hip [Nm/kg]', fontsize=20)
# ax3.set_xlabel('Time [ms]', fontsize=20)

# ax1.plot(time, (ankle_moment[start:end]/subj_mass), color='black')
# ax1.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax1.axes.fill_between(time, ankle_moment[start:end]/subj_mass,where=ankle_power[start:end] >= 0,interpolate=True,color='b',alpha=0.2, label='concentric')
# ax1.axes.fill_between(time, ankle_moment[start:end]/subj_mass,where=ankle_power[start:end] <= 0,interpolate=True,color='r',alpha=0.2, label='eccentric')
# ax1.legend()
# ax1.plot(time, np.zeros([len(ankle_moment[start:end])]), color='gray',linestyle='dashed')
# ax2.plot(time, knee_moment[start:end]/subj_mass, color='black')
# ax2.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax2.axes.fill_between(time, knee_moment[start:end]/subj_mass,where=knee_power[start:end] >= 0,interpolate=True,color='b',alpha=0.2)
# ax2.axes.fill_between(time, knee_moment[start:end]/subj_mass,where=knee_power[start:end] <= 0,interpolate=True,color='r',alpha=0.2)
# ax2.plot(time, np.zeros([len(ankle_moment[start:end])]), color='gray',linestyle='dashed')
# ax3.plot(time, (hip_moment[start:end]/subj_mass)*-1, color='black')
# ax3.axes.fill_between(time, hip_moment[start:end]/subj_mass*-1,where=hip_power[start:end] <= 0,interpolate=True,color='b',alpha=0.2)
# ax3.axes.fill_between(time, hip_moment[start:end]/subj_mass*-1,where=hip_power[start:end] >= 0,interpolate=True,color='r',alpha=0.2)
# ax3.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax3.plot(time, np.zeros([len(ankle_moment[start:end])]), color='gray',linestyle='dashed')
# ax3.set_ylim(-5,7)
# ### Power ###
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
# ax1.plot(time, (ankle_power[start:end]/subj_mass), color='black')
# ax1.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax2.plot(time, knee_power[start:end]/subj_mass, color='black')
# ax2.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax3.plot(time, (hip_power[start:end]/subj_mass)*-1, color='black')
# ax3.axvline(x=0, color='gray', linestyle='dashed', label='ground contact')
# ax1.set_title('Joint Power', fontsize=30)
# ax1.set_ylabel('Ankle [W/kg]', fontsize=20)
# ax2.set_ylabel('Knee [W/kg]', fontsize=20)
# ax3.set_ylabel('Hip [W/kg]', fontsize=20)
# ax3.set_xlabel('Time [ms]', fontsize=20)
# ax3.set_ylim(-55,22)
# ### Forces unfiltered ###
# fig, (ax1) = plt.subplots(nrows=1, ncols=1)
# ax1.plot(data_force['Fz'][200:1250]/subj_mass, color='black')
# ax1.plot(data_force['Fy'][200:1250]/subj_mass, color='blue')
