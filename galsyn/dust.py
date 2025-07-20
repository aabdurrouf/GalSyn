import sys
import numpy as np
from galsyn import config 

def tau_dust_given_z(z, norm_dust_z, norm_dust_tau):
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

def bump_strength_from_dust_index(dust_index):
    # Based on Kriek & Conroy (2013)
    return 0.85 - 1.9*dust_index

def modified_calzetti_dust_curve(AV, wave, dust_index=0.0, Bump_strength=None):
    wave = wave/1e+4     # in micron
    idx = np.where(wave <= 0.63)[0]
    k_lambda1 = 4.05 + (2.659*(-2.156 + (1.509/wave[idx]) - (0.198/wave[idx]/wave[idx]) + (0.011/wave[idx]/wave[idx]/wave[idx])))

    idx = np.where(wave > 0.63)[0]
    k_lambda2 = 4.05 + (2.659*(-1.857 + (1.040/wave[idx]))) 

    k_lambda = k_lambda1.tolist() + k_lambda2.tolist()
    k_lambda = np.asarray(k_lambda)

    if Bump_strength is None:
        Bump_strength = bump_strength_from_dust_index(dust_index)
        
    D_lambda = drude_profile(Bump_strength, wave)

    wave_V = 0.5500
    A_lambda = AV*(k_lambda + D_lambda)*np.power(wave/wave_V, dust_index)/4.05

    return A_lambda

def unresolved_dust_birth_cloud(AV, wave, dust_index_bc=-0.7):
    wave_V = 5500.0
    A_lambda = AV*np.power(wave/wave_V, dust_index_bc)
    return A_lambda

    


