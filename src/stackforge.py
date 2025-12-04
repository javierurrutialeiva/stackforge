import matplotlib.pyplot as plt
import numpy as np
import illustris_python as il
from astropy import constants as const
from astropy import units as u
from astropy import constants as consts
from matplotlib.colors import LogNorm 
from scipy.spatial import cKDTree
import importlib
import time 

import emcee
import corner

from multiprocessing import Pool, Value, shared_memory, Manager, cpu_count
import sys

try:
    from astropy.cosmology import Planck18_arXiv_v2 as cosmo
except:
    from astropy.cosmology import Planck18 as cosmo
    
from tqdm import tqdm
from scipy.optimize import curve_fit
import h5py

from helpers import *
 

class halo():
    def __init__(self, sim = "TNG", **kwargs):
        if sim == "TNG" or sim == "illustris":
            basePath = kwargs.get("basePath",None)
            if basePath is None:
                raise TypeError("BasePath must be an string if sim='TNG' or 'illustris', not None.")
            snapNum = kwargs.get("snapNum",99) if sim == 'illustris' else kwargs.get("snapNum", 135)
            redshift = kwargs.get("redshift", None)         
            if redshift is not None:
                d = load_redshifts(basePath) 
                closest_redshift = np.argmin(np.abs(np.array(list(d.values())) - redshift))
                snapNum = int(list(d.keys())[closest_redshift])
                
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
            self.cBoxSize = BoxSize
            self.BoxSize = BoxSize/(redshift + 1)
            
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
        elif sim == "empty":
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
                           comoving = False):
        if snap is None:
            snap = il.snapshot.loadSubset(self.basePath, self.snapNum, 'gas', 
                    fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])
        self.generate_profiles(R, projection, store = True, comoving = comoving)
        profiles1h = self.profiles3D if projection == "3d" else self.profiles2D
        self.generate_profiles(R, projection, store = True, use_snap = True, 
                               snap = snap, triax = triax, comoving = comoving)
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
    def add_property(self, key, units, volume_weighted = False, 
                     average = False, mass_weighted = False, 
                     projected = False):
        if hasattr(self, "properties") == False:
            self.properties = {}
        if type(key) == str:
            property_dict = {}
            property_dict["name"] = key 
            property_dict["volume_weighted"] = volume_weighted
            property_dict["average"] = average
            property_dict["mass_weighted"] = mass_weighted
            property_dict["projected"] = projected
            uns = get_variables(units)
            uns_dict = {unit:getattr(u,unit) for unit in uns}
            property_units =  eval(units, {}, uns_dict)
            property_dict["units"] = property_units
            self.properties[key] = property_dict
        elif isinstance(key, (np.ndarray, list, tuple)):
            for i,k in enumerate(key):
                units_i = units[i]
                property_dict = {}
                property_dict["name"] = k 
                property_dict["volume_weighted"] = volume_weighted[i] if isinstance(volume_weighted, (np.ndarray, list, tuple)) else volume_weighted
                property_dict["average"] = average[i] if isinstance(average, (np.ndarray, list, tuple)) else average
                property_dict["mass_weighted"] = mass_weighted[i] if isinstance(mass_weighted, (np.ndarray, list, tuple)) else mass_weighted
                property_dict["projected"] = projected[i] if isinstance(projected, (np.ndarray, list, tuple)) else projected
                uns = get_variables(units_i)
                uns_dict = {unit:getattr(u,unit) for unit in uns}
                property_units =  eval(units_i, {}, uns_dict)
                property_dict["units"] = property_units
                self.properties[k] = property_dict
        isinstance(key, (np.ndarray, list, tuple))
    def add_definition(self, definition, volume_weighted = False, average = False, mass_weighted = False, projected = False):
        if hasattr(self, "definitions") == False:
            self.definitions = {}
        if type(definition) is str:
            expr_dict = parse_expression(definiton)
            name = expr_dict["name"]
            expr_dict["volume_weighted"] = volume_weighted
            expr_dict["average"] = average
            expr_dict["mass_weighted"] = mass_weighted
            expr_dict["projected"] = projected
            self.definitions[name] = expr_dict
        if isinstance(definition, (np.ndarray, list, tuple)):
            for d in definition:
                expr_dict = parse_expression(definition)
                name = expr_dict["name"]
                expr_dict["volume_weighted"] = volume_weighted[i] if isinstance(volume_weighted, (np.ndarray, list, tuple)) else volume_weighted
                expr_dict["average"] = average[i] if isinstance(average, (np.ndarray, list, tuple)) else average
                expr_dict["mass_weighted"] = mass_weighted[i] if isinstance(mass_weighted, (np.ndarray, list, tuple)) else mass_weighted
                expr_dict["projected"] = projected[i] if isinstance(projected, (np.ndarray, list, tuple)) else projected
                self.definitions[name] = expr_dict                
    def add_constant(self, const, value = None, units = None):
        if hasattr(self, "constants") == False:
            self.constants = {}
        if type(const) is str:
            if value is None:
                c = getattr(consts, const)
                c_dict = dict(
                    const_name = const,
                    value = c.value,
                    units = c.unit
                )
                self.constants[const] = c_dict
            else:
                uns = get_variables(units)
                uns_dict = {unit:getattr(u,unit) for unit in uns}
                const_units =  eval(units, {}, uns_dict)
                c_dict = dict(
                    const_name = const,
                    value = float(value),
                    units = const_units
                )
                self.constants[const] = c_dict
        elif isinstance(const, (np.ndarray, list, tuple)):
            for i,c in enumerate(const):
                if value is None:
                    c_dict = dict(
                        const_name = c,
                        value = getattr(consts, c).value,
                        units = getattr(consts, c).unit
                    )
                    
                else:
                    if value[i] is not None:
                        uns = get_variables(units[i])
                        uns_dict = {unit:getattr(u,unit) for unit in uns}
                        const_units =  eval(units[i], {}, uns_dict)
                        c_dict = dict(
                            const_name = c,
                            value = float(value[i]),
                            units = const_units
                        )
                    elif value[i] is None:
                        c_dict = dict(
                            const_name = c,
                            value = getattr(consts, c).value,
                            units = getattr(consts, c).unit
                        )  
                self.constants[c] = c_dict
                
    def generate_profiles(self, R, projection="3d", use_snap=False,
                          store=True, snap=None, comoving=False,
                          triax=False, dtype = np.float32,
                          chunk_size=int(1e7), use_multiprocessing = False,
                          ncores = 4,
                          partType = "gas",
                          axis = ["x","y"],
                          sub = False,
                          R_spacing = "linear", 
                          expr_mode = False, root = "functions"):
        if expr_mode == False:
            if use_multiprocessing == True:
                n_cores_total = cpu_count()
                print(f"Using {ncores} of {n_cores_total} available cores!")
            R = np.asarray(R)
            r_edges = R
            n_bins = len(r_edges) - 1
            if n_bins <= 0:
                raise ValueError("R must contain bin edges (len(R) >= 2).")
            if not use_snap:
                gas = self.gas
            elif use_snap and snap is None:

                gas = il.snapshot.loadSubset(basePath, snapNum, 'gas',
                                             fields=["Coordinates", "InternalEnergy", "ElectronAbundance", "Masses"])
            else:
                gas = snap

            h = self.h
            redshift = self.redshift
            coords = gas['Coordinates']        # (N,3)
            Mass = gas['Masses']            # (N,)
            Masses = Mass*1e10 if use_snap == True else Mass
            u_part = gas['InternalEnergy']    # (N,)
            e_abundance = gas['ElectronAbundance']  # (N,)
            if sub == True:
                pos = self.Pos
                fields = gas.keys()
                results = {}
                stored = 0
                if projection == "3d":
                    valid = compute_mask(coords, pos, R.max(), "sphere")
                elif projection == "2d":
                    ax1, ax2 = axis
                    i1 = {'x': 0, 'y': 1, 'z': 2}[ax1]
                    i2 = {'x': 0, 'y': 1, 'z': 2}[ax2]
                    pos = np.asarray([pos[i1], pos[i2]])
                    coords = np.column_stack([coords[:,i1], coords[:,i2]])
                    valid = compute_mask_2d(coords, pos, R.max(), "circle")
                idx = np.where(valid)[0]
                total_valid = len(idx)
                for field in fields:
                    c = 1*int(field == "Coordinates")*int(projection == "2d")
                    field_shape = (total_valid,) + tuple(np.array(gas[field].shape[1:]) - c)
                    field_dtype = gas[field].dtype
                    results[field] = np.empty(field_shape, dtype = field_dtype)
                    if field == "Coordinates":
                        extract_2d(coords, idx, results[field], stored)
                    else:
                        extract_1d(gas[field], idx, results[field], stored)
            N = coords.shape[0]
            if N == 0:
                raise ValueError("No gas particles found in snapshot.")

            coords = coords.astype(dtype, copy=False)
            Masses = Masses.astype(dtype, copy=True)
            u_part = u_part.astype(dtype, copy=False)
            e_abundance = e_abundance.astype(dtype, copy=False)

            R_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

            if projection == "3d":

                vol_shells = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3) 

                factor = 1.0 if comoving else 1.0 / (1.0 + redshift)
                pos = np.asarray(self.Pos, dtype=dtype)

                if use_multiprocessing == False:

                    mass_sum, count, E_sum, U_sum = _worker_numba(r_edges, pos, coords, 
                                                                  Masses, e_abundance, u_part, 
                                                                  chunk_size, R_spacing)
                else:
                    pool = Pool(ncores)
                    _coords = np.array_split(coords, ncores, axis = 0)
                    _Masses = np.array_split(Masses, ncores)
                    _e_abundance = np.array_split(e_abundance, ncores)
                    _u_part = np.array_split(u_part, ncores)
                    _res = []
                    manager = Manager()
                    counter = manager.Value("i", 0)
                    total = int((len(Masses)/chunk_size))
                    for i in range(ncores):
                        kwds = dict(
                                r_edges = r_edges,
                                pos = pos,
                                coords = _coords[i],
                                Masses = _Masses[i],
                                e_abundance = _e_abundance[i],
                                u_part = _u_part[i],
                                chunk_size = chunk_size,
                                factor = factor,
                                counter = counter,
                                total = total,
                                worker_id = i,
                                min_size = min_size, 
                                ntree = ntree,
                                use_tree = use_tree
                            )
                        _res.append(pool.apply_async(_worker, kwds = kwds))
                    res = [r.get() for r in _res]
                    pool.join()
                    pool.close()
                    manager.shutdown() 
                    mass_sum, count, E_sum, U_sum = np.sum(res, axis = 0)

                nonzero = count > 0
                E_mean = np.zeros_like(E_sum)
                U_mean = np.zeros_like(U_sum)
                E_mean[nonzero] = E_sum[nonzero] / count[nonzero]
                U_mean[nonzero] = U_sum[nonzero]

                density_raw = mass_sum / vol_shells
                density = ((density_raw * u.Msun / u.kpc**3).to(u.kg / u.m**3)).value

                T = 10.0**(utherm_ne_to_temp(U_mean, E_mean))
                ne = rho_to_ne(E_mean, density)

                tSZ = ne * T
                rSZ = ne * T**2

                if store:
                    self.R_centers = R_centers
                    self.ne_3D = ne
                    self.tSZ_3D = tSZ
                    self.rSZ_3D = rSZ
                    self.T = T
                    self.E = E_mean
                    self.U = U_mean
                    self.profiles3D = np.vstack((ne, tSZ, rSZ, T, E_mean, U_mean))
                    return None
                else:
                    return R_centers, ne, tSZ, rSZ

            elif projection == "2d":
                area_ann = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

                if triax:
                    projections = [('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'z', 'x')]
                    profiles = []
                    for ax1, ax2, axh in projections:
                        i1 = {'x': 0, 'y': 1, 'z': 2}[ax1]
                        i2 = {'x': 0, 'y': 1, 'z': 2}[ax2]

                        mass_sum = np.zeros(n_bins, dtype=np.float64)
                        count = np.zeros(n_bins, dtype=np.int64)
                        E_sum = np.zeros(n_bins, dtype=np.float64)
                        U_sum = np.zeros(n_bins, dtype=np.float64)

                        factor = 1.0 if comoving else 1.0 / (1.0 + redshift)

                        pos = np.asarray(self.Pos, dtype=np.float32)
                        pos = np.asarray([pos[i1], pos[i2]])
                        if use_multiprocessing == False:
                            mass_sum, count, E_sum, U_sum = _worker2d(r_edges, pos, coords, Masses, 
                                                                   e_abundance, u_part, chunk_size, 
                                                                   factor, counter = None, total = None,
                                                                   i1 = i1, i2 = i2, R_spacing = R_spacing)
                        else:
                            pool = Pool(ncores)
                            _coords = np.array_split(coords, ncores, axis = 1)
                            _Masses = np.array_split(Masses, ncores)
                            _e_abundance = np.array_split(e_abundance, ncores)
                            _u_part = np.array_split(u_part, ncores)
                            _res = []
                            manager = Manager()
                            counter = manager.Value("i", 0)
                            total = int((len(Masses)/chunk_size))
                            for i in range(ncores):
                                kwds = dict(
                                        r_edges = r_edges,
                                        pos = pos,
                                        coords = _coords[i],
                                        Masses = _Masses[i],
                                        e_abundance = _e_abundance[i],
                                        u_part = _u_part[i],
                                        chunk_size = chunk_size,
                                        factor = factor,
                                        counter = counter,
                                        total = total,
                                        i1 = i1,
                                        i2 = i2,
                                        min_size = min_size,
                                        ntree = ntree,
                                        use_tree = use_tree
                                    )
                                _res.append(pool.apply_async(_worker2d, kwds = kwds))
                            res = [r.get() for r in _res]
                            pool.join()
                            pool.close()
                            mass_sum, count, E_sum, U_sum = np.sum(res, axis = 0)

                        nonzero = count > 0
                        E_mean = np.zeros_like(E_sum)
                        U_mean = np.zeros_like(U_sum)
                        E_mean[nonzero] = E_sum[nonzero] / count[nonzero]
                        U_mean[nonzero] = U_sum[nonzero] / count[nonzero]

                        density_raw = mass_sum / area_ann
                        density = ((density_raw * u.Msun / u.kpc**2).to(u.kg / u.m**2)).value

                        T = 10.0**(utherm_ne_to_temp(U_mean, E_mean))
                        ne = rho_to_ne(E_mean, density)
                        tSZ = ne * T
                        rSZ = ne * T**2

                        profiles.append((ne, tSZ, rSZ, T, E_mean, U_mean))

                    ne_mean = np.mean([p[0] for p in profiles], axis=0)
                    tSZ_mean = np.mean([p[1] for p in profiles], axis=0)
                    rSZ_mean = np.mean([p[2] for p in profiles], axis=0)
                    T = np.mean([p[3] for p in profiles], axis=0)
                    E_mean = np.mean([p[4] for p in profiles], axis=0)
                    U_mean = np.mean([p[5] for p in profiles], axis=0)

                    if store:
                        self.R_centers = R_centers
                        self.ne_2D = ne_mean
                        self.tSZ_2D = tSZ_mean
                        self.rSZ_2D = rSZ_mean
                        self.T = T
                        self.E = E_mean
                        self.profiles2D = np.vstack((ne_mean, tSZ_mean, rSZ_mean, T, E_mean))
                        return None
                    else:
                        return R_centers, ne_mean, tSZ_mean, rSZ_mean

                else:
                    ax1,ax2 = axis
                    i1 = {'x': 0, 'y': 1, 'z': 2}[ax1]
                    i2 = {'x': 0, 'y': 1, 'z': 2}[ax2]
                    mass_sum = np.zeros(n_bins, dtype=np.float64)
                    count = np.zeros(n_bins, dtype=np.int64)
                    E_sum = np.zeros(n_bins, dtype=np.float64)
                    U_sum = np.zeros(n_bins, dtype=np.float64)

                    factor = 1.0 if comoving else 1.0 / (1.0 + redshift)
                    pos = np.asarray(self.Pos, dtype=np.float32)
                    pos = np.asarray([pos[i1], pos[i2]])
                    if use_multiprocessing == False:
                        mass_sum, count, E_sum, U_sum = _worker2d(r_edges, pos, coords, Masses, 
                                                               e_abundance, u_part, chunk_size, 
                                                               factor, counter = None, total = None,
                                                               i1 = i1, i2 = i2)
                    else:
                        pool = Pool(ncores)
                        _coords = np.array_split(coords, ncores, axis = 1)
                        _Masses = np.array_split(Masses, ncores)
                        _e_abundance = np.array_split(e_abundance, ncores)
                        _u_part = np.array_split(u_part, ncores)
                        _res = []
                        manager = Manager()
                        counter = manager.Value("i", 0)
                        total = int((len(Masses)/chunk_size))
                        for i in range(ncores):
                            kwds = dict(
                                    r_edges = r_edges,
                                    pos = pos,
                                    coords = _coords[i],
                                    Masses = _Masses[i],
                                    e_abundance = _e_abundance[i],
                                    u_part = _u_part[i],
                                    chunk_size = chunk_size,
                                    factor = factor,
                                    counter = counter,
                                    total = total,
                                    i1 = i1,
                                    i2 = i2,
                                    rmax = rmax
                                )
                            _res.append(pool.apply_async(_worker2d, kwds = kwds))
                        res = [r.get() for r in _res]
                        pool.close()
                        pool.join()
                        mass_sum, count, E_sum, U_sum = np.sum(res, axis = 0)

                    nonzero = count > 0
                    E_mean = np.zeros_like(E_sum)
                    U_mean = np.zeros_like(U_sum)
                    E_mean[nonzero] = E_sum[nonzero] / count[nonzero]
                    U_mean[nonzero] = U_sum[nonzero] / count[nonzero]

                    density_raw = mass_sum / area_ann
                    density = ((density_raw * u.Msun / u.kpc**2).to(u.kg / u.m**2)).value

                    T = 10.0**(utherm_ne_to_temp(U_mean, E_mean))
                    ne = rho_to_ne(E_mean, density)
                    tSZ = ne * T
                    rSZ = ne * T**2

                    if store:
                        self.R_centers = R_centers
                        self.ne_2D = ne
                        self.tSZ_2D = tSZ
                        self.rSZ_2D = rSZ
                        self.T = T
                        self.E = E_mean
                        self.profiles2D = np.vstack((ne, tSZ, rSZ, T, E_mean))
                        return None
                    else:
                        return R_centers, ne, tSZ, rSZ

            else:
                raise ValueError("projection must be '3d' or '2d'")
        elif expr_mode == True:
            
            if root is not None:
                root = importlib.import_module(root)
                                    
            R = np.asarray(R)
            r_edges = R
            properties = self.properties
            contants = self.constants if hasattr(self, "constants") else {}
            definitions = self.definitions if hasattr(self, "definitions") else {}
            functions = [d["func"] for d in definitions.values()]
            property_names = np.unique(list(self.properties.keys()))
            fields = property_names
            constants_dict = {}
            if len(constants.keys()) > 0:
                for k in constants.keys():
                    constants_dict[name] = constants[k]["value"]*constants[k]["units"]
            ax1, ax2 = axis
            i1 = {'x': 0, 'y': 1, 'z': 2}[ax1]
            i2 = {'x': 0, 'y': 1, 'z': 2}[ax2]
            
            if 'Coordinates' not in fields:
                fields = np.append(fields, 'Coordinates')            
            if len(functions) > 0:
                funcs_dict = {}
                for f in functions:
                    if callable(f):
                        funcs_dict[f.__name__] = f
                    elif type(f) is str:
                        funcs_dict[f] = getattr(root, f)
            if not use_snap:
                snap = self.gas
            elif use_snap and snap is None:
                snap = il.snapshot.loadSubset(basePath, snapNum, partType,
                                             fields=fields)
           
            vol_shells = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3) 
            area_rings = 2 * np.pi * (r_edges[1:]**2 - r_edges[:-1]**2) 
            R_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            coords = snap['Coordinates']  
            radial_profiles = dict()
            N = coords.shape[0]
            if N == 0:
                raise ValueError("No particles found in snapshot.")
            coords3d = coords.astype(dtype, copy=False)
            coords2d = np.column_stack([coords3d[:,i1], coords3d[:,i2]])
            pos = self.Pos
            pos2d = np.asarray([pos[i1], pos[i2]])
            N = np.shape(coords3d)[0]
            start = 0 
            stop = chunk_size
            results = {p:np.zeros(len(r_edges)-1) for p in property_names}
            counts = np.zeros(len(r_edges) - 1)
            mass_profile = np.zeros(len(r_edges) - 1)
            mass_profile2d = np.zeros(len(r_edges) - 1)
            while start < N:
                rchunk = compute_offsets(coords3d[start:stop], pos)
                rchunk2d = compute_offsets_2d(coords3d[start:stop], pos)
                if R_spacing == "linear":
                    inds = compute_bin_indices_linear(rchunk, r_edges.min(), r_edges.max(), len(r_edges))
                    inds2d = compute_bin_indices_linear(rchunk2d, r_edges.min(), r_edges.max(), len(r_edges))
                elif R_spacing == "log":
                    inds = compute_bin_indices_log(rchunk, r_edges.min(), r_edges.max(), len(r_edges))
                    inds2d = compute_bin_indices_log(rchunk2d, r_edges.min(), r_edges.max(), len(r_edges))
                    
                attach_numba(snap["Masses"]*1e10, inds, mass_profile, len(r_edges) - 1)
                attach_numba(snap["Masses"]*1e10, inds2d, mass_profile2d, len(r_edges) - 1)
                for i,field in enumerate(property_names): 
                    if i == 0:
                        count_numba(snap[field], inds, counts, len(r_edges) - 1) 
                    prop = properties[field]
                    if prop["mass_weighted"] == True:
                        if prop["projected"] == False:
                            attach_numba(snap[field], inds, results[field], len(r_edges) - 1, snap["Masses"])
                        else: 
                            attach_numba(snap[field], inds2d, results[field], len(r_edges) - 1, snap["Masses"])
                    else:
                        if prop["projected"] == False:
                            attach_numba(snap[field], inds, results[field], len(r_edges) - 1)
                        else:
                            attach_numba(snap[field], inds2d, results[field], len(r_edges) - 1)                       
                start+=chunk_size
                stop+=chunk_size            
            for i, field in enumerate(property_names):
                prop = properties[field]
                if prop["volume_weighted"] == True:
                    if prop["projected"] == False:
                        results[field] = results[field]/vol_shells
                    else:
                        results[field] = results[field]/area_rings
                if prop["average"] == True:
                    results[field] = results[field]/counts
                if prop["mass_weighted"] == True:
                    if prop["projected"] == False:
                        results[field] = results[field]/mass_profile
                    else:
                        results[field] = results[field]/mass_profile2d
            
            if len(definitions.keys()) > 0:
                for k in definitions.keys():
                    di = definition[k]
                    if di['type'] == 'function':
                        func = funcs_dict[di['function']]
                        args = [results[arg_i] for arg_i in funcs_dict['args']]
                        res = func(*args)
                        results[k] = res
                    elif di['type'] == 'operation':
                        opr = di['operation']
                        var = results | constants_dict
                        res = eval(opr, {}, var)
                        results[k] = res
            return results
            if defs is not None:
                output_dict = {}
                for var in defs_dict.keys():
                    args = defs_dict[var]["args"]
                    func = defs_dict[var]["func"]
                    p = [properties[a] for a in args if a != 'Coordinates']
                    if root is not None:
                        output_dict[var] = getattr(root, func)(*p)
                    else:
                        output_dict[var] = funcs_dict[func](*p)
            else:
                output_dict = properties
            for k in properties.keys():
                if k in variables:
                    output_dict[k] = properties[k]
            output_dict["V"] = vol_shells
            output_dict["N"] = counts
            result = eval(expr, {}, output_dict)
            return result
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
            self.cBoxSize = BoxSize
            self.BoxSize = BoxSize/(redshift + 1)

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
                try: 
                    h = halo(sim, basePath = basePath, 
                             snapNum = snapNum, 
                             haloID = ID, 
                             load_subHalo = load_subHalo)
                except KeyError:
                    continue
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
        if sim != "empty":
            self.halos = halos
         
    @classmethod
    def load_halos(cls, profiles1h, profiles2h = None, masses = None, **kwargs):
        n_halos = len(profiles1h)
        if profiles2h is not None:
            if len(profiles2h) != n_halos:
                raise ValueError("profiles1h and profiles2h must have the same length")
        if masses is not None:
            if len(masses) != n_halos:
                raise ValueError("masses and profiles2h must have the same length")
        for key, values in kwargs.items():
            if len(values) != n_halos:
                raise ValueError(f"Parameter '{key}' must have the same length as profiles1h")
        halos = []
        for i in range(n_halos):
            halo_i = halo(sim = "empty")
            halo_i.profiles1h = profiles1h[i]
            if profiles2h is not None:
                halo_i.profiles2h = profiles2h[i]
            if masses is not None:
                halo_i.Mass = masses[i]
            halo_i.group = {}
            for key, values in kwargs.items():
                halo_i.group[key] = values[i]
            halos.append(halo_i)
        out = cls(sim = "empty")
        out.halos = halos
        return out
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
    def compute_profiles(self, R, projection = "3d", triax = False, 
                             snap = None, mean_subtracted = False,
                             progressBar = True, comoving = False, 
                             two_halo_term = True):              
        if snap is None and hasattr(self, "snap"):
            snap = self.snap
        elif snap is None and hasattr(self,"snap") == False:
            snap =  il.snapshot.loadSubset(basePath, snapNum, 'gas', 
                                fields = ["Coordinates","InternalEnergy","ElectronAbundance","Masses"])   
        if np.shape(snap['Coordinates'])[1] == 2:
            projection = "2d"
        it = range(len(self.halos))
        if progressBar:
            pbar = tqdm(total = len(self.halos), desc="Processing halos")
           
        for i in it:
            h = self.halos[i]
            if two_halo_term == True:
                h.separate_into_1h2h(R, projection, triax = triax, snap = snap, 
                                     comoving = comoving)
            else:
                h.generate_profiles(R, projection, triax = triax, comoving = comoving)
            if progressBar == True:
                pbar.update(1)
    @classmethod
    def selection(cls, sim = "TNG", Mmin = 10, Mmax = 15, **kwargs): 
        if sim == "TNG" or sim == "illustris":
            basePath = kwargs.get("basePath", None)
            snapNum = kwargs.get("snapNum", 99) if sim == "TNG" else kwargs.get("snapNum", 135)
            load_only_wsub = kwargs.get("load_only_wsub", True)
            redshift = kwargs.get("redshift", None)
            if redshift is not None:
                d = load_redshifts(basePath) 
                closest_redshift = np.argmin(np.abs(np.array(list(d.values())) - redshift))
                snapNum = int(list(d.keys())[closest_redshift])
            cat = il.groupcat.loadHalos(basePath, snapNum, fields = ['GroupFirstSub', 'GroupMass'])
            group_mass = cat["GroupMass"]
            group_firstsub = cat["GroupFirstSub"]
            logM = np.log10(group_mass * 1e10)
            mask = (logM >= Mmin) & (logM <= Mmax)
            if load_only_wsub:
                mask &= (group_firstsub != -1)
            haloIDs = np.where(mask)[0]
            new_cat = dict(
                GroupMass = group_mass[mask],
                GroupFirstSub = group_firstsub[mask]
            )
            return haloIDs, new_cat
    
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
            
            keys = list(self.halos[0].group.keys())
            for k in keys:
                f.create_dataset(k, data = np.array([h.group[k] for h in self.halos]))
    def bootstrapping(self, nsamples=1000, one_halo = True, two_halo = False, total = False):
        output = []
        if one_halo == True:
            profiles1h = np.array([h.profiles1h for h in self.halos]) 
            if profiles1h.ndim == 3:
                boot1h = Nhalos = profiles1h.shape[0]
                for i in range(nsamples):
                    resample_idx = np.random.randint(0, Nhalos, Nhalos)
                    boot1h[i] = np.mean(profiles1h[resample_idx], axis=0)                
                mu1h = np.mean(boot1h, axis = 0)
                boot1h_flat = boot1h.reshape(nsamples, -1)
                cov1h = np.cov(boot1h_flat, rowvar = False)
                output.append(cov1h)
        if two_halo == True:
            profiles2h = np.array([h.profiles2h for h in self.halos]) 
            Nhalos = profiles2h.shape[0]
            if profiles2h.ndim == 3:
                
                boot2h = np.zeros((nsamples, profiles2h.shape[1], profiles2h.shape[2]))

                for i in range(nsamples):
                    resample_idx = np.random.randint(0, Nhalos, Nhalos)
                    boot2h[i] = np.mean(profiles2h[resample_idx], axis=0)

                mu2h = np.mean(boot2h, axis=0)

                boot2h_flat = boot2h.reshape(nsamples, -1)

                cov2h = np.cov(boot2h_flat, rowvar=False)
                output.append(cov2h)
        if total == True:
            profiles1h = np.array([h.profiles1h for h in self.halos]) 
            profiles2h = np.array([h.profiles2h for h in self.halos]) 
            profilesT = profiles1h + profiles2h
            Nhalos = profilesT.shape[0]
            if profiles2h.ndim == 3:
                
                bootT = np.zeros((nsamples, profilesT.shape[1], profilesT.shape[2]))

                for i in range(nsamples):
                    resample_idx = np.random.randint(0, Nhalos, Nhalos)
                    bootT[i] = np.mean(profilesT[resample_idx], axis=0)

                muT = np.mean(bootT, axis=0)

                bootT_flat = bootT.reshape(nsamples, -1)

                covT = np.cov(bootT_flat, rowvar=False)
                output.append(covT)
        return output
        
class sampler():
    def __init__(self, x, y, cov = None, root = 'functions'):
        cov = np.eye(len(x)) if cov is None else cov
        self.x = x
        self.y = y
        self.cov = cov
        self.root = root
    def generate_priors(self, priors):
        prior_funcs = []
        prior_args = []
        prior_behaviors = []
        fixed_params = []
        free_params = []
        labels = []
        for i,p in enumerate(priors):
            label, behavior, func, args = p.split("|")
            args = np.array(args.split(","), dtype = float) if behavior == "free" else float(args)
            labels.append(label)
            if behavior not in ("free", "fixed"):
                raise TypeError(f"Behavior must be 'free' or fixed. It recieved {behavior}")
            if behavior == "free":
                prior_args.append(args)
                prior_behaviors.append('free')
                free_params.append(i)
                prior_funcs.append(func)
            elif behavior == "fixed":
                fixed_params.append((i, args))
                prior_behaviors.append('fixed')
        self.prior_funcs = prior_funcs
        self.prior_args = prior_args
        self.prior_behaviors = prior_behaviors
        self.fixed_params = fixed_params
        self.free_params = free_params
        self.labels = labels
    def __repr__(self):
        return self.__dict__
    def __str__(self):
        return self.__dict__
    def save(self, folder = None):
        folder = folder if folder is not None else "./"
        if folder != "./" and folder is not None:
            if os.path.exists(folder) == False:
                os.mkdir(folder)
        x = self.x
        y = self.y
        cov = self.cov
        
        self.folder = folder
        
        np.savetxt(f"{folder}/x.txt", x)
        np.savetxt(f"{folder}/y.txt", y)
        np.savetxt(f"{folder}/cov.txt",cov)
    def run(self, model, nwalkers, nsteps, blobs = True, overwrite = False, 
            output_file = "samples.h5", ncores = 1, likelihood = "gaussian",
            initial_guess = None, use_numba = False, demo = False, folder = None):
        root = self.root
        funcs = importlib.import_module(root)
        if hasattr(self, "prior_funcs") == False:
            raise KeyError("'prior_funcs' wasn't found. Run 'generate_priors' first.")
        x = self.x
        y = self.y
        cov = self.cov
        folder = self.folder if hasattr(self, "folder") else folder
        folder = folder + "/" if folder[-1] != "/" else folder
        if folder is not None:
            self.folder = folder
            if os.path.exists(folder) == False and folder != "./":
                os.mkdir(folder)
            output_file = folder + output_file
        if overwrite == True:
            if os.path.exists(output_file):
                os.remove(output_file)
        backend = emcee.backends.HDFBackend(output_file)
        
        self.filename = output_file
        if type(model) == str:
            model = getattr(funcs, model) 
        prior_funcs = self.prior_funcs
        prior_args = self.prior_args
        prior_behaviors = self.prior_behaviors
        fixed_params = self.fixed_params
        free_params = self.free_params
        global ln_prior
        if use_numba == False:
            def ln_prior(theta, prior_behaviors, prior_funcs, prior_args):
                prior = 0.0
                i_theta = 0
                for i in range(len(prior_behaviors)):
                    if prior_behaviors[i] == 'free':
                        args = prior_args[i_theta]
                        func = prior_funcs[i_theta]
                        value = func(theta[i_theta], *args)
                        prior+=value
                        i_theta+=1
                return prior
        else:
            def ln_prior(theta, prior_behaviors, prior_funcs, prior_args):
                prior = 0.0
                i_theta = 0 
                for i in range(len(prior_behaviors)):
                    if prior_behaviors[i] == 'free':
                        args = prior_args[i_theta]
                        func = prior_funcs[i_theta]
                        value = func(theta[i_theta], *args)
                        prior+=value
                        i_theta+=1
                return prior                
        global ln_likelihood
        if use_numba == False:         
            def ln_likelihood(theta, x, y, sigma, **kwargs):
                model = kwargs["model"]
                mu = model(x, theta)
                likelihood = kwargs["likelihood"]
                log_det_C = kwargs.get("log_det_C", 0)
                res = (y - mu)

                if np.ndim(sigma) > 1:
                    inv_cov = kwargs["inv_cov_matrix"]
                    chi2 = np.dot(res.T, np.dot(inv_cov, res))
                else:
                    log_det_C = np.sum(np.log(sigma**2))
                    chi2 = np.sum(res**2/sigma**2)
                if likelihood == 'chi2':
                    ln_lk = -0.5 * chi2
                elif likelihood == 'gaussian':
                    ln_lk = -0.5 * (chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                return ln_lk, chi2, mu
        else:
            def ln_likelihood(theta, x, y, sigma, model, likelihood, log_det_C, inv_cov_matrix):
                x = np.asarray(x)
                y = np.asarray(y)
                sigma = np.asarray(sigma)
                if demo == True:
                    print("  Computing model")
                    t1 = time.time()
                mu = model(x, theta)
                if demo == True:
                    t2 = time.time()
                    print(f"  Took {t2 - t1} seconds to compute the model")
                res = y - mu
                if demo == True:
                    print("  Computing chi2")
                    t1 = time.time()
                if sigma.ndim > 1: 
                    chi2 = 0 
                    for i in range(len(res)):
                        for j in range(len(res)):
                            chi2+=res[i]*res[j]*inv_cov_matrix[i,j]
                elif sigma.ndim == 1 or inv_cov_matrix is None:
                    log_det_C = np.sum(np.log(sigma**2))
                    chi2 = np.sum(res**2 / sigma**2)
                if demo == True:
                    t2 = time.time()
                    print(f"  chi2 has computed in {t1 - t2} seconds")
                if likelihood == 'chi2':
                    ln_lk = -0.5 * chi2
                elif likelihood == 'gaussian':
                    ln_lk = -0.5 * (chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                else:
                    ln_lk = 0.0
                return ln_lk, chi2, mu
        global ln_posterior
        def ln_posterior(theta, x, y, sigma, **kwargs):
            ln_likelihood_func = kwargs["ln_likelihood"]
            ln_prior_func = kwargs["ln_prior"]
            
            fixed_params = kwargs['fixed_params']
            free_params = kwargs['free_params']
            prior_behaviors = kwargs['prior_behaviors']
            prior_funcs = kwargs['prior_funcs']
            prior_args = kwargs['prior_args']
            
            blobs = kwargs['blobs']
            if demo == True:
                print("Merging fixed and free parameters")
                t1 = time.time()
            if len(fixed_params) > 0:
                free_params_indx = free_params
                fixed_params_indx = [p[0] for p in fixed_params]
                fixed_params_values = [p[1] for p in fixed_params]
                new_theta = np.empty(len(fixed_params) + len(free_params))
                new_theta[fixed_params_indx] = fixed_params_values
                new_theta[free_params_indx] = theta
                theta = new_theta
            if demo == True:
                t2 = time.time()
                print(f"Took {t2 - t1} seconds to merge parameters")
            if demo == True:
                print("Computing prior")
                t1 = time.time()
            ln_p = ln_prior(theta, prior_behaviors, prior_funcs, prior_args)
            if demo == True:
                t2 = time.time()
                print(f"Priors has computed in {t2 - t1} seconds")
            if np.isnan(ln_p) or np.isfinite(ln_p) == False:
                ln_p = -np.inf
            if not np.isfinite(ln_p):
                if blobs:
                    return -np.inf, np.nan, np.full_like(y, np.nan)
                return -np.inf
            if demo == True:
                print("Computing likelihood")
                t1 = time.time()
            if use_numba == False:
                ln_lk, chi2, mu = ln_likelihood(theta, x, y, sigma, **kwargs) 
            elif use_numba == True:
                likelihood = kwargs["likelihood"]
                model = kwargs["model"]
                log_det_C = kwargs.get("log_det_C", 0)
                inv_cov_matrix = kwargs.get("inv_cov_matrix", 0)
                ln_lk, chi2, mu = ln_likelihood(theta, x, y, sigma, model, likelihood, log_det_C, inv_cov_matrix)
            if demo == True:
                t2 = time.time()
                print(f"Likelihood has computed in {t2 - t1} seconds")
            if chi2<0:
                ln_lk = -np.inf
                chi2 = np.inf
            if np.isnan(ln_lk) or np.isfinite(ln_lk) == False:
                ln_lk = -np.inf
            ln_pos = ln_lk + ln_p
            if np.isnan(ln_pos) or np.isfinite(ln_pos) == False:
                ln_pos = -np.inf
            if blobs == True:
                return ln_pos, chi2, mu
            else:
                return ln_pos  
        if blobs == True:
            dtype = []
            dtype.append(("chi2", np.dtype((np.float64, 1))))
            dtype.append(("signal", np.dtype((np.float64, len(y)))))
        else:
            dtype = None
        prior_funcs = [getattr(funcs, f) for f in prior_funcs] if use_numba == False else [getattr(funcs, f + "_numba") for f in prior_funcs]
        sampler_kwargs = dict(
            model = model,
            ln_prior = ln_prior,
            ln_likelihood = ln_likelihood,
            fixed_params = fixed_params,
            free_params = free_params,
            prior_behaviors = prior_behaviors,
            prior_funcs = prior_funcs,
            prior_args = prior_args,
            blobs = blobs,
            likelihood = likelihood,
            )
        if np.ndim(cov) > 1:
            log_det_C = np.linalg.slogdet(cov)[1]
            inv_cov_matrix = np.linalg.inv(cov)        
            sampler_kwargs["inv_cov_matrix"] = inv_cov_matrix
            sampler_kwargs["log_det_C"] = log_det_C
        ndims = len(free_params)
        param_limits = [args[-2::] for args in prior_args]
        if initial_guess is None:
            initial_guess = np.zeros((nwalkers, ndims))     
            for i in range(len(param_limits)):
                initial_guess[:,i] = np.array(
                    random_initial_steps(param_limits[i], nwalkers, 
                    distribution = prior_funcs[i],
                    dist_args = prior_args[i],
                    nsamples = 1e3)
                    )
        else:
            initial_guess = np.tile(initial_guess, nwalkers)
            initial_guess = np.reshape(initial_guess, (-1, ndims))
            initial_guess = initial_guess + np.random.normal(size = np.shape(initial_guess))
            for i in range(len(initial_guess)):
                for j in range(len(initial_guess[i])):
                    if initial_guess[i,j] < param_limits[j][0]:
                        initial_guess[i,j] = param_limits[j][0]
                    elif initial_guess [i,j] > param_limits[j][1]:
                        initial_guess[i,j] = param_limits[j][1]
        pool = Pool(ncores)
        EnsembleSampler = emcee.EnsembleSampler(
            nwalkers,
            ndims,
            ln_posterior,
            args=(
                x,
                y,
                cov,
            ),
            kwargs=sampler_kwargs,
            pool = pool,
            backend=backend,
            blobs_dtype = dtype
        ) 
        if demo == True:
            pars = initial_guess[0]
            t1 = time.time()
            ln_posterior(pars, x, y, cov, **sampler_kwargs)
            t2 = time.time()
            print(f"Total time = {t2 - t1} seconds.")
        elif demo == False:
            EnsembleSampler.run_mcmc(initial_guess, nsteps, progress=True, store = True) 
    @classmethod
    def load(cls, folder):
        folder = folder + "/" if folder[-1] != "/" else folder
        x = np.loadtxt(folder + "x.txt")
        y = np.loadtxt(folder + "y.txt")
        cov = np.loadtxt(folder + "cov.txt")
        files = os.listdir(folder)
        out = cls(x, y, cov)
        out.folder = folder
        for f in files:
            if "samples" in f:
                try:
                    chain, blobs = out.load_chain(folder + f, load_blobs = True)
                    out.filename = folder + f
                    return out, chain, blobs
                except:
                    chain = out.load_chain(folder + f, load_blobs = False)
                    out.filename = folder + f
                    return out, chain
        return out
    def load_chain(self, filename = None, load_blobs = True, discard = 0, thin = 1):
        filename = self.filename if hasattr(self, "filename") else filename
        backend = emcee.backends.HDFBackend(filename, read_only = True)
        chain = backend.get_chain(flat = True, discard = discard, thin = thin)
        if load_blobs == False:
            return chain
        else:
            blobs = backend.get_blobs(discard = discard, thin = thin)
            return chain, blobs
    def plot_corner(self, filename = None, sigma_ratio = None, fig = None, smooth = None, 
                    plot_priors = True, discard = 0, thin = 1, **kwargs):
        root = self.root
        funcs = importlib.import_module(root)
        
        labels = self.labels
        ndims = len(labels)
        figsize = kwargs.get("figsize", (5*len(labels), 5*len(labels)))
        fig = plt.figure(figsize = figsize) if fig is None else fig
        chain, blobs = self.load_chain(filename = filename, discard = discard, thin = thin)
        labels = [text2latex(l) for l in labels]
        quantiles = kwargs.get("quantiles", [0.16, 0.5, 0.84])
        levels = kwargs.get("levels", (1-np.exp(-0.5), 1-np.exp(-2)))
        truths = np.median(chain, axis = 0)
        truths_color = kwargs.get("truths_color", "black")
        bins = kwargs.get("bins", 30)
        corner_color = kwargs.get("corner_color", "darkblue")
        fill_contour_kwargs = {'colors': [f'dark{corner_color}', f'light{corner_color}'], 'alpha': 0.1}
        title_fontsize = kwargs.get("title_fontsize", 20)
        title_kwargs = {"fontsize":title_fontsize}
        bins_linewidth = kwargs.get("bins_linewidth", 4)
        bins_kwargs = {"bins":bins, "lw": bins_linewidth}
        corner_plot = corner.corner(
            chain,
            fig = fig,
            labels = labels,
            quantiles = quantiles,
            show_titles = True,
            levels = levels,
            truths = truths,
            truth_color = truths_color,
            plot_density = True,
            no_fill_contours = True,
            plot_datapoints = False,
            bins = bins,
            smooth = smooth,
            smooth1d = smooth,
            fill_contours = True,
            hist_bin_factor= 5,
            n_max_ticks = 5,
            color = corner_color,
            title_kwargs = title_kwargs,
            fill_contour_kwargs = fill_contour_kwargs,
            bins_kwargs = bins_kwargs
        )
        axes = fig.get_axes()
        axes = np.array(axes).reshape((ndims, ndims))
        if sigma_ratio is not None:
            for i in range(ndims):
                for j in range(ndims):
                    pminx, pmedianx, pmaxx = calculate_sigma_intervals(chain[:,j], sigma = sigma_ratio)
                    pminy, pmediany, pmaxy = calculate_sigma_intervals(chain[:,i], sigma = sigma_ratio)
                    if i != j:
                        axes[i][j].set_ylim((pmediany - pminy, pmediany + pmaxy))
                        axes[i][j].set_xlim((pmedianx - pminx, pmedianx + pmaxx))
                    elif i == j:
                        axes[i][j].set_xlim((pmedianx - pminx, pmedianx + pmaxx))
        if plot_priors == True:
            prior_funcs = self.prior_funcs
            prior_args = self.prior_args
            prior_behaviors = self.prior_behaviors
            prior_color = kwargs.get("prior_color", "darkgreen")
            prior_alpha = kwargs.get("prior_alpha", 0.1)
            for i in range(ndims):
                for j in range(i + 1):
                    if i == j:
                        prior, pargs = prior_funcs[i], prior_args[i]
                        if prior is None or pargs is None:
                            continue
                        else:
                            if "uniform" in str(prior) or "flat" in str(prior):
                                continue
                            line = axes[i][j].lines[0]
                            xdata,ydata = np.array(line.get_xdata()), np.array(line.get_ydata())
                            xnew = np.linspace(xdata.min(), xdata.max(), 1000)
                            p = np.exp(np.array([getattr(funcs,prior)(xnew_i, *pargs) for xnew_i in xnew]))
                            p = p / p.max() * ydata.max()
                            axes[i][j].fill_between(xnew, p, color = prior_color, lw = 3, alpha = prior_alpha, edgecolor = "black")

        return fig
def _worker2d(r_edges, pos, coords, Masses, e_abundance, u_part, chunk_size, factor, counter = None, total = 1,
             i1 = 0, i2 = 1, ntree = 5, min_size = 2e4, use_tree = False, R_spacing = "linear"):

    n_bins = len(r_edges) - 1
    start = 0
    mass_sum = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    E_sum = np.zeros(n_bins, dtype=np.float64)
    U_sum = np.zeros(n_bins, dtype=np.float64)
    N = coords.shape[0]
    while start < N:
        stop = int(min(N, start + chunk_size))
        cchunk = coords[start:stop]
        if use_tree == True:
            cchunk, idx = tree(cchunk, pos, min_size, ntree)
            mchunk = Masses[start:stop][idx]
            Euchunk = e_abundance[start:stop][idx]
            Uchunk = u_part[start:stop][idx]
        else:
            mchunk = Masses[start:stop]
            Euchunk = e_abundance[start:stop]
            Uchunk = u_part[start:stop] 
        if np.shape(cchunk)[1] == 3:
            d1 = cchunk[:, i1]
            d2 = cchunk[:, i2]
            cchunk = np.vstack((d1,d2)).T
        rchunk = compute_offsets_2d(cchunk, pos)
        
        inds = np.searchsorted(r_edges, rchunk, side='right') - 1
        valid = (inds >= 0) & (inds < n_bins)
        if np.any(valid):
            inds_v = inds[valid]
            np.add.at(mass_sum, inds_v, mchunk[valid].astype(np.float64))
            np.add.at(count, inds_v, 1)
            np.add.at(E_sum, inds_v, Euchunk[valid].astype(np.float64))
            np.add.at(U_sum, inds_v, Uchunk[valid].astype(np.float64))

        del cchunk, mchunk, Euchunk, Uchunk, rchunk, inds, valid
        start = stop
    return mass_sum, count, E_sum, U_sum


@numba.njit(fastmath=True)
def _worker_numba(r_edges, pos, coords, Masses, e_abundance, u_part, chunk_size, R_spacing = "linear"):
    n_bins = len(r_edges) - 1
    mass_sum = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    E_sum = np.zeros(n_bins, dtype=np.float64)
    U_sum = np.zeros(n_bins, dtype=np.float64)
   
    N = coords.shape[0]
    start = 0
    
    while start < N:
        stop = min(N, start + chunk_size)
        
        cchunk = coords[start:stop]
        mchunk = Masses[start:stop]
        Euchunk = e_abundance[start:stop]
        Uchunk = u_part[start:stop]

        process_chunk_numba(r_edges, pos, cchunk, mchunk, Euchunk, Uchunk,
                           mass_sum, count, E_sum, U_sum, R_spacing)
        
        start = stop
    
    return mass_sum, count, E_sum, U_sum


def _worker(r_edges, pos ,coords, Masses, e_abundance, u_part, chunk_size, factor, 
            counter = None, total = 1, worker_id = 0, ntree = 5, min_size = 2e4, 
            use_tree = False):
    n_bins = len(r_edges) - 1
    start = 0
    mass_sum = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    E_sum = np.zeros(n_bins, dtype=np.float64)
    U_sum = np.zeros(n_bins, dtype=np.float64)
    N = coords.shape[0]
    while start < N:
        stop = int(min(N, start + chunk_size))
        cchunk = coords[start:stop] 
        if use_tree == True:
            cchunk, idx = tree(cchunk, pos, min_size, ntree)
            mchunk = Masses[start:stop][idx]
            Euchunk = e_abundance[start:stop][idx]
            Uchunk = u_part[start:stop][idx]
        else:
            mchunk = Masses[start:stop]
            Euchunk = e_abundance[start:stop]
            Uchunk = u_part[start:stop]
 
        rchunk = compute_offsets(cchunk, pos)
                                 
        inds = np.searchsorted(r_edges, rchunk, side='right') - 1
        valid = (inds >= 0) & (inds < n_bins)
        if np.any(valid):
            inds_v = inds[valid]

            np.add.at(mass_sum, inds_v, mchunk[valid].astype(np.float64))
            np.add.at(count, inds_v, 1)
            np.add.at(E_sum, inds_v, Euchunk[valid].astype(np.float64))
            np.add.at(U_sum, inds_v, Uchunk[valid].astype(np.float64))

        del cchunk, mchunk, Euchunk, Uchunk, rchunk, inds, valid
        start = stop
        
        if counter is not None:
            counter.value+=1
            sys.stdout.write(f"\rProgress: ({counter.value} / {total})")
            sys.stdout.flush()
    if counter is not None:
        print(f"Worker ID = {worker_id} has already finished.\n")
    return mass_sum, count, E_sum, U_sum

def process_subvolume(args):  
    i, j, k, pos, basePath, snapNum, R_subvolume, sub_volume_size, redshift, sim, R, fix_boundary_condition, geometry = args

    while True:
        subvolume, halos = load_sub_volume(basePath, snapNum, 'gas', pos = pos, 
                            boxSize = R_subvolume, load_halos = True, redshift = redshift, 
                            verbose = False, 
                            fields = ['Coordinates','Masses','ElectronAbundance', 'InternalEnergy'], 
                            rmax = sub_volume_size, geometry = geometry, 
                            fix_boundary_condition = fix_boundary_condition, 
                            load_only_halos = load_only_halos)
        s = stack(sim = "TNG", basePath = basePath, haloIDs = halos, redshift=redshift, snap = subvolume)
        s.compute_1h2hprofiles(R, comoving = True, progressBar = False)

        profiles1h_ijk = np.array([h.profiles1h for h in s.halos])
        profiles2h_ijk = np.array([h.profiles2h for h in s.halos])

        print(f"  Subvolume ({i},{j},{k}): {len(halos)} halos processed")
        masses = [h.Mass for h in s.halos]
        m200c = [h.group["M200c"] for h in s.halos]
        m500c = [h.group["M500c"] for h in s.halos]
        MTopHat = [h.group["MTopHat"] for h in s.halos]
        R200c = [h.group["R200c"] for h in s.halos]
        R500c = [h.group["R500c"] for h in s.halos]
        RTopHat = [h.group["RTopHat"] for h in s.halos]
        del subvolume, halos, s
        gc.collect()
        return (i, j, k, profiles1h_ijk, profiles2h_ijk, masses,
               m200c, m500c, MTopHat, R200c, R500c, RTopHat)           
def process_subarea(args):
    i, j, k, basePath, snapNum, R_subvolume, sub_volume_size, redshift, sim, R, fix_boundary_condition, geometry, two_halo_term, axis = args
    subarea, halos = load_sub_area(basePath, snapNum, 'gas', pos = pos, boxSize = R_subvolume, load_halos = True,
                                  redshift = redshift, fields = ['Coordinates','Masses','ElectronAbundance', 'InternalEnergy'],
                                  rmax = sub_volume_size, geometry = geometry, fix_boundary_condition = fix_boundary_condition,
                                  axis = axis)
    s = stack(sim = "TNG", basePath = basePath, haloIDs = halos, redshift=redshift, snap = subarea)
    s.compute_1h2hprofiles(R, comoving = True, progressBar = False, projection = "2d")

    profiles1h_ijk = np.array([h.profiles1h for h in s.halos])
    profiles2h_ijk = np.array([h.profiles2h for h in s.halos])

    print(f"  Subvolume ({i},{j},{k}): {len(halos)} halos processed")
    masses = [h.Mass for h in s.halos]
    m200c = [h.group["M200c"] for h in s.halos]
    m500c = [h.group["M500c"] for h in s.halos]
    MTopHat = [h.group["MTopHat"] for h in s.halos]
    R200c = [h.group["R200c"] for h in s.halos]
    R500c = [h.group["R500c"] for h in s.halos]
    RTopHat = [h.group["RTopHat"] for h in s.halos]
    del subarea, halos, s
    gc.collect()
    return (i, j, k, profiles1h_ijk, profiles2h_ijk, masses,
           m200c, m500c, MTopHat, R200c, R500c, RTopHat) 

def process_snapshot(sim = "TNG", basePath = None, R = None, snapNum = None, redshift = None,
                     n_subvolumes = 100, parallel = False, ncores = 1, nmax = None, fix_boundary_condition = False,
                     geometry = "sphere", projection = "3d", axis = ["x","y"], two_halo_term = True):
    if redshift is not None:
        d = load_redshifts(basePath) 
        closest_redshift = np.argmin(np.abs(np.array(list(d.values())) - redshift))
        snapNum = int(list(d.keys())[closest_redshift])  
    
    header = il.groupcat.loadHeader(basePath, snapNum)
    boxSize = header["BoxSize"]
    
    dx = np.linspace(0, boxSize, n_subvolumes, endpoint=False)
    dy = np.linspace(0, boxSize, n_subvolumes, endpoint=False)
    dz = np.linspace(0, boxSize, n_subvolumes, endpoint=False)
    sub_volume_size = dx[1] - dx[0]
    dx += sub_volume_size / 2
    dy += sub_volume_size / 2
    dz += sub_volume_size / 2
    R_subvolume = 1.5*np.max(R) + sub_volume_size
    
    profiles1h = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)
    profiles2h = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)
    masses = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)
    ms200c = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)
    ms500c = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)
    mstophat = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object)             
    rs200c = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object) 
    rs500c = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object) 
    rstophat = np.empty((n_subvolumes, n_subvolumes, n_subvolumes), dtype=object) 
    
    tasks = []
    for i in range(n_subvolumes):
        for j in range(n_subvolumes):
            for k in range(n_subvolumes):
                pos = np.array([dx[i], dy[j], dz[k]])
                if projection == "3d":
                    tasks.append((i, j, k, pos, basePath, snapNum, R_subvolume, 
                            sub_volume_size, redshift, sim, R, fix_boundary_condition,
                            geometry))
                elif projection == "2d":
                    tasks.append((i, j, k, pos, basePath, snapNum, R_subvolume, 
                            sub_volume_size, redshift, sim, R, fix_boundary_condition,
                            geometry, axis))  
                    
    print(f"\nProcessing {len(tasks)} subvolumes...\n")
    if nmax is not None and nmax < len(tasks):
        tasks = tasks[:nmax]
        print(f"\nLimited to first {nmax} subvolumes (out of {n_subvolumes**3} total)")
    
    if projection == "3d":
        if parallel and ncores > 1:
            with Pool(processes=ncores) as pool:
                results = pool.map(process_subvolume, tasks)
        else:
            results = []
            for idx, task in enumerate(tasks):
                print(f"Processing subvolume {idx+1}/{len(tasks)}")
                results.append(process_subvolume(task))
    elif projection == "2d":
        if parallel and ncores > 1:
            with Pool(processes=ncores) as pool:
                results = pool.map(process_subarea, tasks)            
        else:
            results = []
            for idx, task in enumerate(tasks):
                print(f"Processing subvolume {idx+1}/{len(tasks)}")
                results.append(process_subarea(task))
        
    total_halos = 0
    empty_subvolumes = 0
    
    for i, j, k, prof1h, prof2h, mass, m200, m500, mtophat, r200, r500, rtophat,  in results:
        profiles1h[i, j, k] = prof1h
        profiles2h[i, j, k] = prof2h
                   
        masses[i, j, k] = np.array(mass)
        ms200c[i, j, k] = np.array(m200)
        ms500c[i, j, k] = np.array(m500)
        mstophat[i, j, k] = np.array(mtophat)
        rs200c[i, j, k] = np.array(r200)
        rs500c[i, j, k] = np.array(r500)
        rstophat[i, j, k] = np.array(rtophat)
                   
        n_halos = len(prof1h) if len(prof1h) > 0 else 0
        total_halos += n_halos
        if n_halos == 0:
            empty_subvolumes += 1
    profiles1h_flatten = flatten_profiles(profiles1h)
    profiles2h_flatten = flatten_profiles(profiles2h)

    masses_flatten = flatten_halo_properties(masses)
    m200c_flatten = flatten_halo_properties(ms200c)
    m500c_flatten = flatten_halo_properties(ms500c)
    mtophat_flatten = flatten_halo_properties(mstophat)
    r200c_flatten = flatten_halo_properties(rs200c)
    r500c_flatten = flatten_halo_properties(rs500c)   
    rtophat_flatten = flatten_halo_properties(rstophat)
                   
    return (profiles1h_flatten, profiles2h_flatten, masses_flatten, m200c_flatten, 
           m500c_flatten, mtophat_flatten, r200c_flatten, r500c_flatten, 
           rtophat_flatten)

def process_halos(args):
    task_id, R, basePath, snapNum, projection, axis, R_spacing, idx = args

    profiles1h = np.zeros((len(idx), 6, len(R) - 1))
    masses = np.zeros(len(idx))
    ms200c = np.zeros(len(idx))
    ms500c = np.zeros(len(idx))
    mstophat = np.zeros(len(idx))
    rs200c = np.zeros(len(idx))
    rs500c = np.zeros(len(idx))
    rstophat = np.zeros(len(idx))
    
    for i in range(len(idx)):
        haloID = idx[i]
        try: 
            hi = halo(sim = "TNG", basePath = basePath, snapNum = snapNum, haloID = haloID) 
            hi.generate_profiles(R, projection = projection, triax = False, axis = axis, R_spacing = R_spacing)
            profiles1h[i] = hi.profiles3D if projection == "3d" else hi.profiles2D
        except KeyError:
            profiles1h[i] = np.full((6 ,len(R) - 1), np.nan)
        masses[i] = hi.Mass
        ms200c[i] = hi.group["M200c"]
        ms500c[i] = hi.group["M500c"]
        mstophat[i] = hi.group["MTopHat"]
        rs200c[i] = hi.group["R200c"]
        rs500c[i] = hi.group["R500c"]
        rstophat[i] = hi.group["RTopHat"]
    print(f" ({task_id}) => {len(idx)} halos have been processed.")
    return task_id, profiles1h, masses, ms200c, ms500c, mstophat, rs200c, rs500c, rstophat

def extract_one_halo_term(sim = "TNG", basePath = None, R = None, snapNum = None, parallel = True, redshift = None,
                          ncores = 1, nchunks = 20, Mmin = None, Mmax = None, projection = "3d", 
                          axis = ["x","y"], R_spacing = "linear"):
    
    if redshift is not None:
        d = load_redshifts(basePath) 
        closest_redshift = np.argmin(np.abs(np.array(list(d.values())) - redshift))
        snapNum = int(list(d.keys())[closest_redshift])  
    print("Extracting one halo term from snapshot",snapNum)
    header = il.groupcat.loadHeader(basePath, snapNum)
    boxSize = header["BoxSize"]
    print("Loading all available halo")
    all_available_halos = il.groupcat.loadHalos('../sims.TNG/TNG300-3/output/', 99, 
                                  fields = ['GroupMass'])
    masses = all_available_halos*1e10
    print("Available halos =", len(masses))
    idx = np.arange(0., len(masses), 1, dtype = np.int32)
    if Mmin is not None:
        print(f"  Minimum halo mass set to {Mmin}")
        mask = np.where(masses >= Mmin)
        idx = idx[mask]
        masses = masses[mask]
        print(f"  New halo lenght = {len(masses)}")
    if Mmax is not None:
        print(f"  Maximum halo mass set to {Mmax}")
        mask = np.where(masses <= Mmax)
        idx = idx[mask]
        masses = masses[mask]
        print(f"  New halo lenght = {len(masses)}")
    idx = np.array_split(idx, nchunks)
    
    profiles1h = np.empty(len(idx), dtype = object)
    masses = np.empty(len(idx), dtype = object)
    ms200c = np.empty(len(idx), dtype = object)
    ms500c = np.empty(len(idx), dtype = object)
    mstophat = np.empty(len(idx), dtype = object)
    rs200c = np.empty(len(idx), dtype = object)
    rs500c = np.empty(len(idx), dtype = object)
    rstophat = np.empty(len(idx), dtype = object)
    
    tasks = []
    for i,idx_i in enumerate(idx):
        tasks.append([i, R, basePath, snapNum, projection, axis, R_spacing, idx_i])
    if parallel == False or ncores == 1:
        results = []
        for i in range(len(tasks)):
            print(f"  Currently running task {i} with {len(tasks[i][-1])} halos.")
            results.append(process_halos(tasks[i]))
    elif parallel == True and ncores > 1:
        with Pool(processes=ncores) as pool:
            results = pool.map(process_halos, tasks)
    total_halos = 0 
    for i, prof1h, mass, m200, m500, mtophat, r200, r500, rtophat in results:
        profiles1h[i] = np.asarray(prof1h)
        masses[i] = np.asarray(mass)
        ms200c[i] = np.asarray(m200)
        ms500c[i] = np.asarray(m500)
        mstophat[i] = np.asarray(mtophat)
        rs200c[i] = np.asarray(r200)
        rs500c[i] = np.asarray(r500)
        rstophat[i] = np.asarray(rtophat)
                   
        n_halos = len(prof1h) if len(prof1h) > 0 else 0
        total_halos += n_halos
    print(f"Total processed halos = {total_halos}")
    
    profiles1h_flatten = flatten_profiles_2d(profiles1h, total_halos, 6, len(R) - 1)
    masses_flatten = flatten_scalars(masses, total_halos)
    ms200c_flatten = flatten_scalars(ms200c, total_halos)
    ms500c_flatten = flatten_scalars(ms500c, total_halos)
    mstophat_flatten = flatten_scalars(mstophat, total_halos)
    rs200c_flatten = flatten_scalars(rs200c, total_halos)
    rs500c_flatten = flatten_scalars(rs500c, total_halos)
    rstophat_flatten = flatten_scalars(rstophat, total_halos)
    
    return (profiles1h_flatten, masses_flatten, ms200c_flatten, ms500c_flatten, 
           mstophat_flatten, rs200c_flatten, rs500c_flatten, rstophat_flatten)

