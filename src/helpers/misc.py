import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il
import numba 

from astropy import constants as const
from astropy import units as u
import sympy as sp
import re


#=====numba functions======
@numba.njit(fastmath = True)
def _flatten_profiles(profiles_3d, output, total_halos, n_bins):
    shape = profiles_3d.shape
    current_row = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                prof = profiles_3d[i, j, k]
                if prof is not None and len(prof) > 0:
                    n_halos_in_subvol = prof.shape[0]
                    for h in range(n_halos_in_subvol):
                        for b in range(n_bins):
                            output[current_row, b] = prof[h, b]
                        current_row += 1
    
    return output
@numba.njit(fastmath=True)
def searchsorted_numba(edges, values):
    n = len(values)
    inds = np.empty(n, dtype=np.int32)   
    for i in range(n):
        val = values[i]
        left = 0
        right = len(edges)
        while left < right:
            mid = (left + right) // 2
            if edges[mid] < val:
                left = mid + 1
            else:
                right = mid
        inds[i] = left - 1
    return inds

@numba.njit(fastmath = True, cache = True)
def process_chunk_numba(r_edges, pos, coords, Masses, e_abundance, u_part,
                        mass_sum, count, E_sum, U_sum, R_spacing = "linear"):
    n_bins = len(r_edges) - 1
    rchunk = compute_offsets(coords, pos)
    if R_spacing == "linear":
        inds = compute_bin_indices_linear(rchunk, r_edges.min(), r_edges.max(), len(r_edges))
    elif R_spacing == "log":
        inds = compute_bin_indices_log(rchunk, r_edges.min(), r_edges.max(), len(r_edges))
    for i in range(len(inds)):
        bin_idx = inds[i]
        if bin_idx >= 0 and bin_idx < n_bins:
            mass_sum[bin_idx] += Masses[i]
            count[bin_idx] += 1
            E_sum[bin_idx] += e_abundance[i]
            U_sum[bin_idx] += u_part[i]


@numba.njit(fastmath=True)
def compute_bin_indices_linear(r_values, r_min, r_max, n_bins):

    N = len(r_values)
    indices = np.empty(N, dtype=np.int32)
    bin_width = (r_max - r_min) / n_bins
    
    for i in range(N):
        r = r_values[i]
        if r < r_min or r >= r_max:
            indices[i] = -1
        else:
            bin_idx = int((r - r_min) / bin_width)
            indices[i] = min(bin_idx, n_bins - 1)
    
    return indices

@numba.njit(fastmath=True)
def compute_bin_indices_log(r_values, r_min, r_max, n_bins):
    N = len(r_values)
    indices = np.empty(N, dtype=np.int32)
    
    log_r_min = np.log10(r_min)
    log_r_max = np.log10(r_max)
    log_bin_width = (log_r_max - log_r_min) / n_bins
    
    for i in range(N):
        r = r_values[i]
        if r < r_min or r >= r_max:
            indices[i] = -1
        else:
            log_r = np.log10(r)
            bin_idx = int((log_r - log_r_min) / log_bin_width)
            indices[i] = min(bin_idx, n_bins - 1)
    
    return indices

@numba.njit(fastmath = True)
def attach_numba(var, inds, output, counts, n_bins):
    for i in range(len(inds)):
        bin_idx = inds[i]
        if bin_idx >= 0 and bin_idx < n_bins:
            counts[bin_idx]+=1
            output[bin_idx]+=var[i]
            
@numba.njit(fastmath=True)
def compute_offsets(coords, pos):
    N = coords.shape[0]
    offset = np.empty(N, dtype = np.float32)
    for i in numba.prange(N):
        dx = coords[i, 0] - pos[0]
        dy = coords[i, 1] - pos[1]
        dz = coords[i, 2] - pos[2]
        offset[i] = (dx*dx + dy*dy + dz*dz)**0.5
    return offset
 
@numba.njit(fastmath=True)
def compute_offsets_2d(coords, pos):
    N = coords.shape[0]
    offset = np.empty(N, dtype = np.float32)
    for i in numba.prange(N):
        dx = coords[i, 0] - pos[0]
        dy = coords[i, 1] - pos[1]
        offset[i] = (dx*dx + dy*dy)**0.5
    return offset

@numba.njit(fastmath=True)
def compute_mask_2d(coords, pos, boxSize, shape="circle"):
    N = coords.shape[0]
    valid = np.empty(N, dtype=np.bool_)
    
    pos_x = pos[0]
    pos_y = pos[1]
    
    for i in range(N):
        dx = coords[i, 0] - pos_x
        dy = coords[i, 1] - pos_y
        
        if shape == "circle":
            r2 = dx*dx + dy*dy
            valid[i] = (r2 <= boxSize * boxSize)
        elif shape == "box":
            cond_x = np.abs(dx) <= boxSize/2.
            cond_y = np.abs(dy) <= boxSize/2.
            valid[i] = cond_x and cond_y
    
    return valid

@numba.njit(fastmath=True, cache = True)
def compute_mask(coords, pos, boxSize, shape="sphere"):
    N = coords.shape[0]
    valid = np.empty(N, dtype=np.bool_)

    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]
    rmax = boxSize*boxSize
    half = boxSize/2.
    for i in numba.prange(N): 
        dx = coords[i, 0] - pos_x
        dy = coords[i, 1] - pos_y
        dz = coords[i, 2] - pos_z
        
        if shape == "sphere":
            r2 = dx*dx + dy*dy + dz*dz
            valid[i] = (r2 <= rmax)
        elif shape == "box":
            cond_x = np.abs(dx) <= half
            cond_y = np.abs(dy) <= half
            cond_z = np.abs(dz) <= half
            valid[i] = cond_x and cond_y and cond_z
    
    return valid

@numba.njit()
def extract_1d(source, indices, target, offset):
    n = len(indices)
    for i in numba.prange(n):
        target[offset + i] = source[indices[i]]


@numba.njit()
def extract_2d(source, indices, target, offset):
    n = len(indices)
    m = source.shape[1]
    for i in numba.prange(n):
        for j in range(m):
            target[offset + i, j] = source[indices[i], j]


@numba.njit()
def extract_3d(source, indices, target, offset):
    n = len(indices)
    m = source.shape[1]
    k = source.shape[2]
    for i in numba.prange(n):
        for j in range(m):
            for l in range(k):
                target[offset + i, j, l] = source[indices[i], j, l]

@numba.njit(fastmath=True)
def periodic_boundary_condition_fix_coordinates(halo_pos, coordinates_to_fix, box_size):
    n = len(coordinates_to_fix)
    half_box = 0.5 * box_size
    
    pos_x, pos_y, pos_z = halo_pos[0], halo_pos[1], halo_pos[2]
    for i in numba.prange(n):

        if coordinates_to_fix[i, 0] < pos_x - half_box:
            coordinates_to_fix[i, 0] += box_size
        elif coordinates_to_fix[i, 0] > pos_x + half_box:
            coordinates_to_fix[i, 0] -= box_size
        

        if coordinates_to_fix[i, 1] < pos_y - half_box:
            coordinates_to_fix[i, 1] += box_size
        elif coordinates_to_fix[i, 1] > pos_y + half_box:
            coordinates_to_fix[i, 1] -= box_size
        if np.shape(coordinates_to_fix)[1] == 3:
            if coordinates_to_fix[i, 2] < pos_z - half_box:
                coordinates_to_fix[i, 2] += box_size
            elif coordinates_to_fix[i, 2] > pos_z + half_box:
                coordinates_to_fix[i, 2] -= box_size

    return coordinates_to_fix

#=================

            
def compute_radial_bin_indices(r_edges, pos, coords, chunk_size=None):
    N = coords.shape[0]
    
    if chunk_size is None or chunk_size >= N:
        radii = compute_offsets(coords, pos)
        bin_indices = searchsorted_numba(r_edges, radii)
    else:
        radii = np.empty(N, dtype=np.float32)
        bin_indices = np.empty(N, dtype=np.int32)
        
        start = 0
        while start < N:
            stop = min(N, start + chunk_size)
            chunk_coords = coords[start:stop]
            chunk_radii = compute_offsets(chunk_coords, pos)
            chunk_bins = searchsorted_numba(r_edges, chunk_radii)
            
            radii[start:stop] = chunk_radii
            bin_indices[start:stop] = chunk_bins
            start = stop
    n_bins = len(r_edges) - 1
    bin_indices[bin_indices >= n_bins] = -1
    bin_indices[bin_indices < 0] = -1
    
    return bin_indices


def load_chunk_data_fast(f, gName, field, idx, result, stored):
    if isinstance(f, np.ndarray) == True:
        data_full = f
    else:
        dataset = f[gName][field]
        data_full = dataset[:]
    n_valid = len(idx)
    ndim = len(data_full.shape)
    
    if ndim == 1:
        extract_1d(data_full, idx, result, stored)
    elif ndim == 2:
        extract_2d(data_full, idx, result, stored)
    elif ndim == 3:
        extract_3d(data_full, idx, result, stored)
    else:
        result[stored:stored + n_valid] = data_full[idx]
    del data_full

class Found_Error_Config(Exception):
    pass

def _as_f32(a):
    return a.astype(np.float32, copy=False)

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
        try:
            header = il.groupcat.loadHeader(basePath, ni)
            redshift = header["Redshift"]
            d[str(ni)] = redshift
        except:
            continue
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

def tree(cchunk, pos, min_size = 2e4, nmax=20, shrink_factor=0.5):
    idx = np.arange(len(cchunk))
    mins = cchunk.min(axis=0)
    maxs = cchunk.max(axis=0)
    size = maxs - mins
    n = 0

    while np.any(size > min_size) and n < nmax:
        half_new = np.maximum(0.5 * min_size, 0.5 * size * shrink_factor)
        mins_new = pos - half_new
        maxs_new = pos + half_new

        mask = np.all((cchunk >= mins_new) & (cchunk <= maxs_new), axis=1)
        idx = idx[mask]
        cchunk = cchunk[mask]

        if len(cchunk) == 0:
            break
        mins, maxs = cchunk.min(axis=0), cchunk.max(axis=0)
        size = maxs - mins
        n += 1

    return cchunk, idx

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
def flatten_halo_properties(properties_3d):
    property_list = []
    
    shape = properties_3d.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                prop = properties_3d[i, j, k]
                if prop is not None and isinstance(prop, np.ndarray) and len(prop) > 0:
                    if prop.ndim > 1:
                        prop = prop.flatten()
                    property_list.append(prop)
    
    if len(property_list) == 0:
        print("Warning: No properties found to flatten")
        return np.array([])

    properties_flat = np.concatenate(property_list)
    
    return properties_flat

def get_variables(expr):
    variables = re.findall(r'[A-Za-z_]\w*', expr)
    return variables

def get_operations(expr):
    ops = re.findall(r'[\+\-\*\/\^]', expr)
    return ops

def get_volume_ops(expr):
    result = []
    for m in re.finditer(r'([*/])\s*V', expr):
        op = m.group(1)
        idx = m.start(1)       # index of the operator
        if op == '/':
            result.append(("divide", idx))
        elif op == '*':
            result.append(("multiply", idx))
    return result

def get_def(expr):   
    pattern = r'(\w+)\s*=>\s*(\w+)\((.*?)\)'
    matches = re.findall(pattern, expr)
    vars_dict = {}
    for var, func, args in matches:
        args = [a.strip() for a in args.split(",")]
        vars_dict[var] = dict(func = func, args = args)
    return vars_dict

def random_initial_steps(limits, n, distribution = "uniform", ln_distribution = True,
                nsamples = 1e4, dist_args = None):
    if distribution == "uniform":
        lower, upper = limits
        params = np.random.uniform(lower, upper, size = n)
    elif distribution == "fixed":
        params = np.full(n, limits)
    else:
        assert nsamples > n, "nsamples must be greater than n"
        x = np.linspace(limits[0], limits[1], int(nsamples))
        weights = np.exp([distribution(xi, *dist_args) for xi in x]) if ln_distribution else [distribution(xi, *dist_args) for xi in x]
        weights = np.nan_to_num(weights, np.nanmin(weights))
        weights = weights/np.nansum(weights)
        params = np.random.choice(x, p = weights, size = n)
    return params

def pte(chi2, cov, cinv=None, n_samples=10000, return_samples=False, return_realizations = False):
    if len(cov.shape) == 1:
        cov = np.eye(cov.size) * cov
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    if cinv is None:
        cinv = np.linalg.pinv(cov)
    mc = stats.multivariate_normal(allow_singular = True, cov=cov).rvs(size=n_samples)
    chi2_mc = np.array([np.dot(i, np.dot(cinv, i)) for i in mc])
    pte = (chi2_mc > chi2).sum() / n_samples
    if return_samples == False and return_realizations == False:
        return pte
    else:
        output = [pte]
        if return_samples == True:
            output.append(chi2_mc)
        if return_realizations == True:
            output.append(mc)
        return output



def flatten_profiles(profiles_3d):
    total_halos = 0
    n_bins = None
    profile_list = []
    
    shape = profiles_3d.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                prof = profiles_3d[i, j, k]
                if prof is not None and isinstance(prof, np.ndarray) and len(prof) > 0:
                    if n_bins is None:
                        n_bins = prof.shape[1] if prof.ndim > 1 else len(prof)
                    profile_list.append(prof)
                    total_halos += prof.shape[0] if prof.ndim > 1 else 1
    
    if total_halos == 0:
        print("Warning: No profiles found to flatten")
        return np.array([])

    profiles_flat = np.vstack(profile_list)
    
    return profiles_flat