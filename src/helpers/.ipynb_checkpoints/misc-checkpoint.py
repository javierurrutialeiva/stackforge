import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il

from astropy import constants as const
from astropy import units as u

class Found_Error_Config(Exception):
    pass

def read_dataset(name):
    dset = f[name]
    if dset.shape == ():
        val = dset[()]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return val
    else:
        if dset.dtype.kind in ("S", "O"):
            return np.array([v.decode("utf-8") for v in dset[:]])
        else:
            return dset[:]

def load_redshifts(basePath, nmin = 0, nmax = 99):
    n = np.arange(nmin, nmax + 1)
    d = {}
    for ni in n:
        header = il.groupcat.loadHeader(basePath, ni)
        redshift = header["Redshift"]
        d[str(ni)] = redshift
    return d

def load_snap(basePath, redshift = 0, fields =  
              ["Coordinates","InternalEnergy","ElectronAbundance","Masses"]
             ):
    redshift_dict = load_redshifts(basePath)
    idx = np.argmin(np.abs(np.array(list(redshift_dict.values())) - redshift))
    snapNum = np.array(list(redshift_dict.keys()))[idx]
    snap = il.snapshot.loadSubset(basePath, snapNum, 'gas', 
        fields = fields)
    return snap
def prop2arr(prop,delimiter=',',dtype=np.float64, remove_white_spaces = True):
    """
    convert a property from a configuration file to a numpy array
    """
    arr = prop.replace(' ','').split(delimiter) if remove_white_spaces else prop.split(delimiter)
    return np.array(arr,dtype=dtype)

def str2bool(string):
    return True if string.lower() == "true" else False

def rho_to_ne(nelec, rho):
    """Convert ElectornAbundance and density of gas cells into electron density"""
    hydrogen_massfrac = 0.76
    m_p = const.m_p.value
    ne = nelec*hydrogen_massfrac*rho/m_p
    return ne

def utherm_ne_to_temp(utherm, nelec):
    """ Convert the InternalEnergy and ElectronAbundance of gas cells to temperature in [log K]. """
    hydrogen_massfrac = 0.76 # approximate
    mass_proton = 1.672622e-24 # cgs
    gamma = 5/3
    boltzmann = 1.380650e-16 # cgs (erg/K)

    # unit system
    UnitLength_in_cm = 3.085678e21   # 1.0 kpc
    UnitMass_in_g = 1.989e43 # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0e5 # 1 km/sec

    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    UnitEnergy_in_cgs = UnitMass_in_g * UnitLength_in_cm**2.0 / UnitTime_in_s**2.0

    # calculate mean molecular weight
    meanmolwt = 4.0/(1.0 + 3.0 * hydrogen_massfrac + 4.0* hydrogen_massfrac * nelec)
    meanmolwt *= mass_proton

    # calculate temperature (K)
    temp = utherm * (gamma-1.0) / boltzmann * UnitEnergy_in_cgs / UnitMass_in_g * meanmolwt
    temp = np.log10(temp)

    return temp.astype('float32')
