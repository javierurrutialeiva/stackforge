import matplotlib.pyplot as plt
import numpy as np
import illustris_python as il
from astropy import constants as const
from astropy import units as u
from matplotlib.colors import LogNorm 
from astropy.cosmology import Planck18 as cosmo
from tqdm import tqdm
from scipy.optimize import curve_fit
import h5py

from helpers import *
 
class halo():
    def __init__(self, sim = "TNG", **kwargs):
        if sim == "TNG" or "illustris":
            basePath = kwargs.get("basePath",None)
            if basePath is None:
                raise TypeError("BasePath must be an string if sim='TNG' or 'illustris', not None.")
            snapNum = kwargs.get("snapNum",99) if sim == 'illustris' else kwargs.get("snapNum", 135)
            haloID = kwargs.get("haloID", 0)
            load_subHalo = kwargs.get("load_subHalo", False)
            
            #load simulation in illustris-TNG/illustris way using illustris_python
            
            header = il.groupcat.loadHeader(basePath, snapNum)
            h = header["HubbleParam"]
            redshift = header["Redshift"]
            BoxSize = header["BoxSize"]
            self.sim = sim
            
            self.basePath = basePath
            self.haloID = haloID
            self.snapNum = snapNum
            self.h = h
            
            self.redshift = redshift
            self.cBoxSize = BoxSize/h
            self.BoxSize = BoxSize/h/(redshift + 1)
            
            if load_subHalo == False:
                group = il.groupcat.loadSingle(basePath,snapNum,haloID=haloID)
                self.MassKey = "GroupMass"
                self.PosKey = "GroupPos"
            else:
                group = il.groupcat.loadSingle(basePath,snapNum,subhaloID=haloID)
                self.MassKey = "SubhaloMass"
                self.PosKey = "SubhaloPos"

            self.Pos = group[self.PosKey]
            self.Mass = group[self.MassKey]

            self.group = dict(
                BHMass = group["GroupBHMass"] * 1e10,
                BHMdot = group["GroupBHMdot"] * 1e10,
                GasMetalFraction = group["GroupGasMetalFractions"],
                GasMetallicity = group["GroupGasMetallicity"],
                SNR = group["GroupSFR"],
                StarMetallicity = group["GroupStarMetallicity"],
                Vel = group["GroupVel"] * 1000,
                M200c = group["Group_M_Crit200"] * 1e10,
                M500c = group["Group_M_Crit500"] * 1e10,
                M200m = group["Group_M_Mean200"] * 1e10,
                MTopHat = group["Group_M_TopHat200"] * 1e10,
                R200c = group["Group_R_Crit200"],
                R500c = group["Group_R_Crit500"],
                R200m = group["Group_R_Mean200"],
                RTopHat = group["Group_R_TopHat200"]
            )

            fields = ["Coordinates", "ElectronAbundance", "InternalEnergy", "Masses"]
            
            if not load_subHalo:
                gas = il.snapshot.loadHalo(basePath, snapNum, haloID, 'gas', fields=fields)
            else:
                gas = il.snapshot.loadSubhalo(basePath, snapNum, haloID, 'gas', fields=fields)

            self.gas = dict(
                Coordinates = gas["Coordinates"],
                ElectronAbundance = gas["ElectronAbundance"],
                InternalEnergy = gas["InternalEnergy"],
                Masses = gas["Masses"] * 1e10
            )

        elif sim == "simba":
            pass
    def utherm_ne_to_temp(self, utherm = None, nelec = None, store = False):
        utherm = self.gas['InternalEnergy'] if utherm is None else utherm 
        nelec = self.gas['ElectronAbundance'] if nelec is None else nelec
        
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        UnitEnergy_in_cgs = UnitMass_in_g * UnitLength_in_cm**2.0 / UnitTime_in_s**2.0

        meanmolwt = 4.0/(1.0 + 3.0 * hydrogen_massfrac + 4.0* hydrogen_massfrac * nelec)
        meanmolwt *= mass_proton

        temp = utherm * (gamma-1.0) / boltzmann * UnitEnergy_in_cgs / UnitMass_in_g * meanmolwt
        if store == True:
            self.gas["T"] = temp
        else:
            return temp
    def rho_to_ne(self, nelec = None, rho = None):
        """Convert ElectornAbundance and density of gas cells into electron density"""
        nelec = self.gas['ElectronAbundance'] if nelec is None else nelec
        density = self.gas["Density"] if rho is None else rho
        hydrogen_massfrac = 0.76
        mass_proton = const.m_p.value
        ne = nelec*hydrogen_massfrac*rho/mass_proton
        return ne
    def separate_into_1h2h(self, R, projection = "3d", triax = False, 
                           snap = None, store = True, mean = 0.,
                           comoving = False, remove_h = True, 
                           use_area = False):
        if snap is None:
            snap = il.snapshot.loadSubset(self.basePath, self.snapNum, 'gas', 
                    fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])
        self.generate_profiles(R, projection, store = True, comoving = comoving, remove_h = remove_h
                              ,use_area = use_area)
        profiles1h = self.profiles3D if projection == "3d" else self.profiles2D
        self.generate_profiles(R, projection, store = True, use_snap = True, 
                               snap = snap, triax = triax, comoving = comoving, remove_h = remove_h
                               ,use_area = use_area)
        profilesTotal = self.profiles3D if projection == "3d" else self.profiles2D
        profiles2h = profilesTotal - profiles1h
        if store == True:
            self.profiles1h = profiles1h 
            self.profiles2h = profiles2h 
        else:
            Rbins, profiles1h, profiles2h 
    def compute_mass(self, fb = 0.155, projection = "3d"):
        """Compute enclosed mass from electron number density assuming f_b = Omega_b/Omega_m = 0.155"""
        if projection == "3d":
            ne = self.profiles3D[0]
            electronAbundance = self.profiles3D[4]
            R = self.R_centers 
            Xh = 0.76
            density = ne/(Xh*electronAbundance)*const.m_p.value
            gas_mass = np.trapz(density*R**2*4*np.pi, x = R)
            total_mass = gas_mass/fb
        elif projection == "2d":
            ne = self.profiles2D[0]
            electronAbundance = self.profiles2D[4]
            R = self.R_centers 
            Xh = 0.76
            density = ne/(Xh*electronAbundance)*const.m_p.value
            gas_mass = np.trapz(density*R*2*np.pi, x = R)
            total_mass = gas_mass/fb 
        return gas_mass, total_mass
    def generate_profiles(self, R, projection = "3d", use_snap = False, 
                          store = True, snap = None,  comoving = False, 
                          triax = False, method = 'histogram',
                          remove_h = True, use_area = False):
        if use_snap == False:
            gas = self.gas 
        elif use_snap == True and snap is None:
            gas = il.snapshot.loadSubset(basePath, snapNum, 'gas', 
                                fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])
        else:
            gas = snap
            
        group = self.group
        h = self.h
        redshift = self.redshift
        
        Masses = gas['Masses']/h if remove_h == True else gas['Masses']
        u_part = gas["InternalEnergy"]
        e_abundance = gas['ElectronAbundance']
        Temp = 10**utherm_ne_to_temp(u_part, e_abundance) 
        if projection == "3d":

            dx = gas['Coordinates'][:,0] - self.Pos[0]
            dy = gas['Coordinates'][:,1] - self.Pos[1]
            dz = gas['Coordinates'][:,2] - self.Pos[2]
            
            factor = 1 if comoving == True else 1/(1 + redshift)
            factor /= h if remove_h == True else 1
            dx, dy, dz = dx*factor, dy*factor, dz*factor 
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            del dx, dy, dz   
            b3D = binner3D(r, Rbins = R)
            R_centers, density, counts = b3D.apply(Masses, statistic="vol", method = method)
            density = ((density*u.Msun/u.kpc**3).to(u.kg/u.m**3))
            R_centers, E, _ = b3D.apply(e_abundance, statistic="mean", method = method)
            R_centers, T, _ = b3D.apply(Temp, statistic="mean", method = method)
            del b3D, Masses, e_abundance, Temp, u_part
            ne = self.rho_to_ne(E, density).value
            tSZ = ne*T
            rSZ = const.k_B.value**2*ne*T**2
            
            if store == True:
                self.R_centers = R_centers
                self.ne_3D = ne
                self.tSZ_3D = tSZ
                self.rSZ_3D = rSZ
                self.T = T
                self.E = E
                self.profiles3D = np.vstack((ne, tSZ, rSZ, T, E))
            else:
                return R_centers, ne, tSZ, rSZ
            
        elif projection == "2d":
            if triax == False:
                dx = gas['Coordinates'][:,0] - self.Pos[0]
                dy = gas['Coordinates'][:,1] - self.Pos[1]
                dz = gas['Coordinates'][:,2] - self.Pos[2]
                
                factor = 1 if comoving == True else 1/(1 + redshift)
                factor /= h if remove_h == True else 1
                dx, dy, dz = dx*factor, dy*factor, dz*factor 

                r = np.sqrt(dx**2 + dy**2)

                if use_snap == False and use_area == False:
                    height = dh.max() - dh.min()
                elif use_snap == True and use_area == False:
                    height = self.BoxSize
                elif use_area == True:
                    height = 1.0
                b2D = binner2D(r, Rbins = R, height = height)
                R_centers, density, counts = b2D.apply(Masses, statistic="vol", method = method)
                density = ((density*u.Msun/u.kpc**3).to(u.kg/u.m**3))
                
                R_centers, U, _ = b2D.apply(u_part, statistic="mean", method = method)
                R_centers, E, _ = b2D.apply(e_abundance, statistic="mean", method = method)
                T = 10**utherm_ne_to_temp(U, E)
                ne = rho_to_ne(E, density).value
                tSZ = ne*T
                rSZ = ne*T**2 
                if store == True:
                    self.R_centers = R_centers
                    self.ne_2D = ne
                    self.tSZ_2D = tSZ
                    self.rSZ_2D = rSZ
                    self.T = T
                    self.E = E
                    self.profiles2D = np.vstack((ne, tSZ, rSZ, T, E))
                else:
                    return R_centers, ne, tSZ, rSZ
            elif triax == True:
                profiles = []
                projections = [('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'z', 'x')]  
                for ax1, ax2, axh in projections:
                    i1 = {'x': 0, 'y': 1, 'z': 2}[ax1]
                    i2 = {'x': 0, 'y': 1, 'z': 2}[ax2]
                    ih = {'x': 0, 'y': 1, 'z': 2}[axh]

                    d1 = gas['Coordinates'][:, i1] - self.Pos[i1]
                    d2 = gas['Coordinates'][:, i2] - self.Pos[i2]
                    dh = gas['Coordinates'][:, ih] - self.Pos[ih]
                    
                    factor = 1 if comoving == True else 1/(1 + redshift)
                    factor /= h if remove_h == True else 1
                    d1, d2, dh = d1*factor, d1*factor, dh*factor 

                    r = np.sqrt(d1**2 + d2**2)

                    if use_snap == False and use_area == False:
                        height = dh.max() - dh.min()
                    elif use_snap == True and use_area == False:
                        height = self.BoxSize
                    elif use_area == True:
                        height = 1.0
                    b2D = binner2D(r, Rbins=R, height=height)

                    R_centers, density, counts = b2D.apply(Masses, statistic="vol", method = method)
                    density = (density * u.Msun / u.kpc**3).to(u.kg / u.m**3)
                
                    R_centers, U, _ = b2D.apply(u_part, statistic="mean", method = method)
                    R_centers, E, _ = b2D.apply(e_abundance, statistic="mean", method = method)
                    T = utherm_ne_to_temp(U, E)
                    ne = rho_to_ne(E, density).value

                    tSZ = ne * T
                    rSZ = ne * T**2

                    profiles.append((ne, tSZ, rSZ))
                    
                ne_mean = np.mean([p[0] for p in profiles], axis=0)
                tSZ_mean = np.mean([p[1] for p in profiles], axis=0)
                rSZ_mean = np.mean([p[2] for p in profiles], axis=0)

                if store:
                    self.R_centers = R_centers
                    self.ne_2D = ne_mean
                    self.tSZ_2D = tSZ_mean
                    self.rSZ_2D = rSZ_mean
                    self.T = T
                    self.E = E
                    self.profiles2D = np.vstack((ne, tSZ, rSZ, T, E))
                else:
                    return R_centers, ne_mean, tSZ_mean, rSZ_mean
class stack():
    def __init__(self, sim = "TNG", Mlogmin = None, Mlogmax = None, redshift = None, **kwargs):
        if sim == "TNG" or sim == "illustris":
            basePath = kwargs.get("basePath", None)
            snapNum = kwargs.get("snapNum", 99) if sim == "TNG" else kwargs.get("snapNum", 135)
            haloIDs = kwargs.get("haloIDs", [0]) 
            load_snap = kwargs.get("load_snap", True)
            load_subHalo = kwargs.get("load_subHalo", False)
            snap = kwargs.get("snap", None)
            
            if redshift is not None:
                d = load_redshifts(basePath) 
                closest_redshift = np.argmin(np.abs(np.array(list(d.values())) - redshift))
                snapNum = int(list(d.keys())[closest_redshift])
            header = il.groupcat.loadHeader(basePath, snapNum)
            h = header["HubbleParam"]
            redshift = header["Redshift"]
            BoxSize = header["BoxSize"]
            self.basePath = basePath
            self.haloIDs = haloIDs
            self.snapNum = snapNum
            self.h = h
            self.redshift = redshift
            self.cBoxSize = BoxSize/h
            self.BoxSize = BoxSize/h/(redshift + 1)

            if load_snap == True:
                if snap is None:
                    snap =  il.snapshot.loadSubset(basePath, snapNum, 'gas', 
                                        fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])
                self.snap = snap
                cmean_density = (((np.sum(snap["Masses"]*1e10)/self.cBoxSize**3)*u.Msun/u.kpc**3).to(u.kg/u.m**3)).value
                mean_electron_abundance = np.mean(snap["ElectronAbundance"])
                cmean_ne = rho_to_ne(mean_electron_abundance, cmean_density)
                self.cmean_ne = cmean_ne
                mean_density = (((np.sum(snap["Masses"]*1e10)/self.BoxSize**3)*u.Msun/u.kpc**3).to(u.kg/u.m**3)).value
                mean_ne = rho_to_ne(mean_electron_abundance, mean_density)
                self.mean_ne = mean_ne
                del snap
            halos = []
            Mmin = 10**Mlogmin if Mlogmin is not None else None
            Mmax = 10**Mlogmax if Mlogmax is not None else None
            for ID in haloIDs:
                h = halo(sim, basePath = basePath, 
                         snapNum = snapNum, 
                         haloID = ID, 
                         load_subHalo = load_subHalo)
                if Mmin is not None:
                    if h.Mass < Mmin:
                        continue
                if Mmax is not None:
                    if h.Mass > Mmax:
                        continue
                if "Masses" in list(h.gas.keys()): 
                    halos.append(h)
        elif sim == "simba":
            pass
        self.halos = halos
    def __str__(self):
        redshift = self.redshift
        basePath = self.basePath
        ids = self.haloIDs
        Mass = [h.Mass for h in self.halos]
        Mmin, Mmax = np.min(Mass), np.max(Mass)
        output = "=="*20
        output+= "\nGroup of halos"
        output+= f"\n*base path = {basePath}"
        output+= f"\n*redshift = {redshift}"
        output+= f"\n*halos IDs [{np.min(ids)},{np.max(ids)}]"
        output+= f"\nN halos = {len(Mass)}"
        output+= f"\nMass [{np.log10(Mmin*1e10/self.h)},{np.log10(Mmax*1e10/self.h)}]"
        return output
    def compute_1h2hprofiles(self, R, projection = "3d", triax = False, 
                             snap = None, mean_subtracted = False,
                             progressBar = True, comoving = False, 
                             remove_h = True):
        if snap is None and hasattr(self, "snap"):
            snap = self.snap
        elif snap is None and hasattr(self,"snap") == False:
            snap =  il.snapshot.loadSubset(basePath, snapNum, 'gas', 
                                fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])           
        it = range(len(self.halos))
        if progressBar:
            pbar = tqdm(total = len(self.halos), desc="Processing halos")

        for i in it:
            h = self.halos[i]
            h.separate_into_1h2h(R, projection, triax = triax, snap = snap, 
                                 comoving = comoving, remove_h = remove_h)
            if progressBar == True:
                pbar.update(1)
    def stack_profiles(self, prop = "ne"):
        if prop == "ne":
            p1h = np.array([h.ne1h for h in self.halos])
            p2h = np.array([h.ne2h for h in self.halos])
            ptotal = p1h + p2h
            stacked_p_total = np.mean(ptotal, axis = 0)
            stacked_1h = np.mean(p1h, axis = 0)
            stacked_2h = stacked_p_total - stacked_1h
            self.stacked_ne1h = stacked_1h
            self.stacked_ne2h = stacked_2h
    @classmethod
    def load(cls, file):
        with h5py.File(file, "r") as f:
            basePath = read_dataset("basePath")
            snapNum = read_dataset("snapNum")
            haloIDs = read_dataset("haloIDs")
            
            out = cls(basePath, snapNum, haloIDs, True)
            for k in f.keys():
                setattr(out, k, read_dataset(k))

    def save(self):
        z = np.round(self.redshift,2)
        profiles1h = [h.profiles1h for h in self.halos]
        profiles2h = [h.profiles2h for h in self.halos]

        with h5py.File(f"profiles_z={z}.h5", "w") as f:
            f.create_dataset("profiles1h", data = profiles1h)
            f.create_dataset("profiles2h", data = profiles2h)
            f.create_dataset("Masses", data = np.array([h.Mass for h in self.halos]))
            f.create_dataset("redshift", data = self.redshift)
            f.create_dataset("h", data = self.h)
            f.create_dataset("basePath", data = self.basePath)
            f.create_dataset("snapNum", data = self.snapNum)
            f.create_dataset("cBoxSize", data = self.cBoxSize)
            f.create_dataset("BoxSize", data = self.BoxSize)
            f.create_dataset("mean_ne", data = self.mean_ne)
            f.create_dataset("cmean_ne", data = self.cmean_ne)
            f.create_dataset("haloIDs", data = self.haloIDs)
    def fit(self, model, method = "lsq", one_halo_only = True, prop = "ne"):
        if prop == "ne":
            if one_halo_only == True:
                P = self.stacked_ne1h
                R = self.halos[0].R_centers
                if method == "lsq":
                    par, cov = curve_fit(model, R, P)
                    return par, np.sqrt(np.diag(cov))
                