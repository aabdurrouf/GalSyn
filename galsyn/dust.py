import sys
import numpy as np
from galsyn import config
import importlib.resources 

wave_V = 0.5500

def tau_dust_given_z(z, norm_dust_z, norm_dust_tau):

    norm_dust_z = np.asarray(norm_dust_z)
    norm_dust_tau = np.asarray(norm_dust_tau)
    
    from scipy.interpolate import interp1d
    
    # based on Vogelsberger+2020 (Table 3)
    f = interp1d(norm_dust_z, norm_dust_tau, fill_value="extrapolate")
    return f(z)

def drude_profile(Bump_strength, wave_um):
    wave0 = 0.2175
    dwave = 0.035
    part1 = wave_um * wave_um * dwave * dwave
    part2 = np.square((wave_um*wave_um) - (wave0*wave0))  

    D_lambda = Bump_strength * part1 / (part2 + part1)
    return D_lambda

def bump_amp_from_dust_index(dust_index):
    # Based on Kriek & Conroy (2013)
    return 0.85 - 1.9*dust_index

def unresolved_dust_birth_cloud_Alambda_per_AV(wave_ang, dust_index_bc=-0.7):
    Alambda_per_AV = np.power(wave_ang/wave_V, dust_index_bc)
    return Alambda_per_AV

def calzetti_dust_klambda(wave_ang):
    wave_um = wave_ang/1e+4     # in micron
    idx = np.where(wave_um <= 0.63)[0]
    k_lambda1 = 4.05 + (2.659*(-2.156 + (1.509/wave_um[idx]) - (0.198/wave_um[idx]/wave_um[idx]) + (0.011/wave_um[idx]/wave_um[idx]/wave_um[idx])))

    idx = np.where(wave_um > 0.63)[0]
    k_lambda2 = 4.05 + (2.659*(-1.857 + (1.040/wave_um[idx]))) 

    k_lambda = k_lambda1.tolist() + k_lambda2.tolist()

    return np.asarray(k_lambda)

def calzetti_dust_Alambda_per_AV(wave_ang):
    k_lambda = calzetti_dust_klambda(wave_ang)
    Alambda_per_AV = k_lambda/4.05
    return Alambda_per_AV

def modified_calzetti_dust_Alambda_per_AV(wave_ang, dust_index=0.0, bump_amp=None):
    k_lambda = calzetti_dust_klambda(wave_ang)

    if bump_amp is None:
        bump_amp = bump_amp_from_dust_index(dust_index)

    wave_um = wave_ang/1e+4
    D_lambda = drude_profile(bump_amp, wave_um)
    Alambda_per_AV = (k_lambda + D_lambda)*np.power(wave_um/wave_V, dust_index)/4.05
    return Alambda_per_AV

def salim18_dust_Alambda_per_AV(wave_ang, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV):
    wave_um = wave_ang/1e+4      # in micron

    D_lambda = drude_profile(salim_B, wave_um)
    k_lambda = salim_a0 + (salim_a1/wave_um) + (salim_a2/wave_um/wave_um) + (salim_a3/wave_um/wave_um/wave_um) + D_lambda + salim_RV

    Alambda_per_AV = k_lambda / salim_RV
    return Alambda_per_AV

def fitzpatrick99_dust_Alambda_per_AV(wave_ang):
    data_file_name = "fitzpatrick99.txt"
    try:
        from scipy.interpolate import interp1d
        data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
        data = np.loadtxt(data_path)
        f = interp1d(data[:,0], data[:,1], fill_value="extrapolate")
        return f(wave_ang)
    
    except Exception as e:
        print(f"Error loading dust normalization data from {data_file_name}: {e}")
        sys.exit(1)

def ccm89_dust_Alambda_per_AV(wave_ang):
    data_file_name = "ccm89.txt"
    try:
        from scipy.interpolate import interp1d
        data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
        data = np.loadtxt(data_path)
        f = interp1d(data[:,0], data[:,1], fill_value="extrapolate")
        return f(wave_ang)
    
    except Exception as e:
        print(f"Error loading dust normalization data from {data_file_name}: {e}")
        sys.exit(1)

def lmc_gordon2003_dust_Alambda_per_AV(wave_ang):
    data_file_name = "lmc_gordon2003.txt"
    try:
        from scipy.interpolate import interp1d
        data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
        data = np.loadtxt(data_path)
        f = interp1d(data[:,0], data[:,1], fill_value="extrapolate")
        return f(wave_ang)
    
    except Exception as e:
        print(f"Error loading dust normalization data from {data_file_name}: {e}")
        sys.exit(1)

def smc_gordon2003_dust_Alambda_per_AV(wave_ang):
    data_file_name = "smc_gordon2003.txt"
    try:
        from scipy.interpolate import interp1d
        data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
        data = np.loadtxt(data_path)
        f = interp1d(data[:,0], data[:,1], fill_value="extrapolate")
        return f(wave_ang)
    
    except Exception as e:
        print(f"Error loading dust normalization data from {data_file_name}: {e}")
        sys.exit(1)


