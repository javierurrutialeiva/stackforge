import h5py
import numpy as np
import os
import illustris_python as il
import gc
from .misc import *
import numba
import time 

    
#some functions were gently stolen from illustris_python

#functions from illustris_python 

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")
    

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart

#---
def load_sub_area(basePath, snapNum=None, partType='gas', axis=['x','y'],
                  pos=np.zeros(3), boxSize=2e4, fields=['Coordinates'],
                  load_halos=True, rmax=5e3, halo_fields=None, verbose=True,
                  redshift=None, fix_boundary_condition=False, 
                  geometry='circle'):
    

    if redshift is not None and snapNum is None:
        redshift_dict = load_redshifts(basePath)
        idx = np.argmin(np.abs(np.array(list(redshift_dict.values())) - redshift))
        snapNum = int(np.array(list(redshift_dict.keys()))[idx])
    
    ptNum = partTypeNum(partType)
    gName = f"PartType{ptNum}"

    if axis[0] == axis[1]:
        raise ValueError("Invalid axis: both axes cannot be the same.")
    

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    i1 = axis_map[axis[0]]
    i2 = axis_map[axis[1]]
  
    pos = np.asarray(pos)
    pos_2d = np.array([pos[i1], pos[i2]])

    if isinstance(fields, str):
        fields = [fields]
    elif isinstance(fields, (list, tuple)):
        fields = list(fields)
        if "Coordinates" not in fields:
            fields.append("Coordinates")
    

    with h5py.File(snapPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        BoxSize = header["BoxSize"]
        nPart = getNumPart(header)
        numToRead = nPart[ptNum]
        
        i = 1
        while gName not in f and i < 1000:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1
        

        if fields is None:
            fields = list(f[gName].keys())
        
        valid_fields = []
        for field in fields:
            if field in f[gName]:
                valid_fields.append(field)
            elif verbose:
                print(f"Warning: Field {field} not found in {gName}, skipping.")
        fields = valid_fields

    fileNum = 0
    total_valid = 0
    field_shapes = {}
    field_dtypes = {}
    cached_masks = {}
    chunk_stats = {}
    
    while True:
        try:
            with h5py.File(snapPath(basePath, snapNum, fileNum), 'r') as f:
                if gName not in f:
                    fileNum += 1
                    continue
                
                if not field_shapes:
                    for field in fields:

                        if field == 'Coordinates':
                            field_shapes[field] = (2,)  
                        else:
                            field_shapes[field] = f[gName][field].shape[1:]
                        field_dtypes[field] = f[gName][field].dtype
                coords_3d = f[gName]['Coordinates'][:]
                if fix_boundary_condition:
                    coords_3d = periodic_boundary_condition_fix_coordinates(
                        pos, coords_3d, BoxSize
                    )
                
                coords_2d = np.column_stack((coords_3d[:, i1], coords_3d[:, i2]))
                
                n_total = len(coords_2d)
               
                valid = compute_mask_2d(coords_2d, pos_2d, boxSize, geometry)
                idx = np.where(valid)[0]
                n_valid = len(idx)
                
                if n_valid > 0:
                    cached_masks[fileNum] = idx
                    total_valid += n_valid
                    chunk_stats[fileNum] = (n_valid, n_total)
                    if verbose:
                        print(f"  Chunk {fileNum}: {n_valid} particles")
                
                del valid, coords_3d, coords_2d
                fileNum += 1
        
        except FileNotFoundError:
            break
    
    if verbose:
        print(f"Total valid particles: {total_valid}")
    
    result = {}
    for field in fields:
        shape = (total_valid,) + field_shapes[field]
        result[field] = np.empty(shape, dtype=field_dtypes[field])
        if verbose:
            print(f"Allocated {field}: {shape}, {result[field].nbytes / 1e9:.2f} GB")

    if verbose:
        print("\nSecond pass: loading data...")
    
    stored = 0
    for fileNum in sorted(cached_masks.keys()):
        idx = cached_masks[fileNum]
        n_valid, n_total = chunk_stats[fileNum]
        
        with h5py.File(snapPath(basePath, snapNum, fileNum), 'r') as f:
            coords_3d = f[gName]['Coordinates'][:]
            
            if fix_boundary_condition:
                coords_3d = periodic_boundary_condition_fix_coordinates(
                    pos, coords_3d, BoxSize
                )

            coords_2d = np.column_stack((coords_3d[:, i1], coords_3d[:, i2]))
            
            # Guardar datos
            for field in fields:
                if field == "Coordinates":
                    extract_2d(coords_2d, idx, result[field], stored)
                else:
                    load_chunk_data_fast(f, gName, field, idx, result[field], stored)
            
            stored += n_valid
            if verbose:
                print(f"  Chunk {fileNum} saved. Progress: {stored}/{total_valid}")
            
            del coords_3d, coords_2d
    
    del cached_masks
    gc.collect()

    if not load_halos:
        return result
    
    if halo_fields is not None:
        if isinstance(halo_fields, str):
            halo_fields = [halo_fields]
        if "GroupPos" not in halo_fields:
            halo_fields.append("GroupPos")
    
    if verbose:
        print("  Extracting halos...")
    
    all_halos = il.groupcat.loadHalos(basePath, snapNum, fields=halo_fields)
    GroupPos_3d = all_halos["GroupPos"]
   
    GroupPos_2d = np.column_stack((GroupPos_3d[:, i1], GroupPos_3d[:, i2]))
    mask = compute_mask_2d(GroupPos_2d, pos_2d, rmax, geometry)
    idx = np.where(mask)[0]
    
    if verbose:
        print(f"  {len(idx)} halos extracted from sub-area!")
    
    return result, idx

@numba.njit(fastmath=True)
def compute_mask_2d(coords, pos, boxSize, shape="sphere", i1 = 0, i2 = 1):
    N = coords.shape[0]
    valid = np.empty(N, dtype=np.bool_)
    
    pos_i1 = pos[0]
    pos_i2 = pos[1]
    
    for i in range(N): 
        if np.shape(coords)[1] == 3:
            di1 = coords[i, i1] - pos_i1
            di2 = coords[i, i2] - pos_i2
        else:
            di1 = coords[i, 0] - pos_i1
            di2 = coords[i, 1] - pos_i2
            
        if shape == "sphere":
            r2 = di1*di1 + di2*di2
            valid[i] = (r2 <= boxSize * boxSize)
        elif shape == "box":
            cond_i1 = np.abs(di1) <= boxSize/2.
            cond_i2 = np.abs(di2) <= boxSize/2.
            valid[i] = cond_i1 and cond_i2
    
    return valid



    
def load_sub_volume(basePath, snapNum = None, partType='gas',
                    pos=np.zeros(3), boxSize=2e4,
                    fields=["Coordinates"], load_halos=True, rmax=5e3,
                    halo_fields = None, verbose = True,
                    redshift = None, fix_boundary_condition = False, 
                    geometry = "sphere", load_only_halos = False):
    if redshift is not None and snapNum is None:
        redshift_dict = load_redshifts(basePath)
        idx = np.argmin(np.abs(np.array(list(redshift_dict.values())) - redshift))
        snapNum = int(np.array(list(redshift_dict.keys()))[idx])
    if load_only_halos == False:
        ptNum = partTypeNum(partType)
        gName = f"PartType{ptNum}"
        if type(fields) is str and fields is not None:
            fields = [fields]
        elif type(fields) is list or type(fields) is tuple:
            fields = list(fields)
            if "Coordinates" not in fields:
                fields.append("Coordinates")

        with h5py.File(snapPath(basePath, snapNum), 'r') as f:
            header = dict(f['Header'].attrs.items())
            BoxSize = header["BoxSize"]
            nPart = getNumPart(header)
            numToRead = nPart[ptNum]

            i = 1
            while gName not in f and i < 1000:
                f = h5py.File(snapPath(basePath, snapNum, i), 'r')
                i += 1

            if fields is None:
                fields = list(f[gName].keys())

            valid_fields = []
            for field in fields:
                if field in f[gName]:
                    valid_fields.append(field)
                else:
                    if verbose:
                        print(f"Warning: Field {field} not found in {gName}, skipping.")
            fields = valid_fields
        fileNum = 0
        total_valid = 0
        field_shapes = {}
        field_dtypes = {}
        cached_masks = {} 
        chunk_stats = {}
        file_handles = []
        while True:
            try:
                f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r') 
                if gName not in f:
                    fileNum += 1
                    continue

                if not field_shapes:
                    if fields is None:
                        fields = list(f[gName].keys())
                    else:
                        valid_fields = [field for field in fields if field in f[gName]]
                        if len(valid_fields) < len(fields):
                            missing = set(fields) - set(valid_fields)
                            if verbose:
                                print(f"Warning: Fields {missing} not found, skipping.")
                        fields = valid_fields

                    for field in fields:
                        field_shapes[field] = f[gName][field].shape[1:]  
                        field_dtypes[field] = f[gName][field].dtype
                coords = f[gName]['Coordinates'][:]
                if fix_boundary_condition == True: 
                    coords = periodic_boundary_condition_fix_coordinates(pos, coords, BoxSize)   
                mins = np.min(coords, axis=0)
                maxs = np.max(coords, axis=0)
                if geometry == "sphere":
                    closest_point = np.clip(pos, mins, maxs)
                    distance = np.linalg.norm(closest_point - pos)
                    overlap = distance <= boxSize/2.
                else:
                    search_min = pos - boxSize/2.
                    search_max = pos + boxSize/2.
                    overlap_x = (mins[0] <= search_max[0]) and (maxs[0] >= search_min[0])
                    overlap_y = (mins[1] <= search_max[1]) and (maxs[1] >= search_min[1])
                    overlap_z = (mins[2] <= search_max[2]) and (maxs[2] >= search_min[2])

                    overlap = overlap_x and overlap_y and overlap_z

                if overlap == False:
                    if verbose == True:
                        print(f"  Skipping chunk {fileNum}, there are no particles inside volume.")
                    fileNum+=1
                    continue                   
                n_total = len(coords)
                valid = compute_mask(coords, pos, boxSize, geometry)
                idx = np.where(valid)[0]
                n_valid = len(idx)

                if n_valid > 0:
                    cached_masks[fileNum] = idx 
                    total_valid += n_valid
                    chunk_stats[fileNum] = (n_valid, n_total)
                    if verbose:
                        print(f"  Chunk {fileNum}: {n_valid} particles")

                del valid, coords
                file_handles.append(f)
                fileNum += 1

            except FileNotFoundError:
                break

        if verbose:
            print(f"Total valid particles: {total_valid}")
        result = {}
        for field in fields:
            shape = (total_valid,) + field_shapes[field]
            result[field] = np.empty(shape, dtype=field_dtypes[field])
            if verbose:
                print(f"Allocated {field}: {shape}, {result[field].nbytes / 1e9:.2f} GB")
        if verbose:
            print("\nSecond pass: loading data...")
        stored = 0
        for fileNum in sorted(cached_masks.keys()):
            idx = cached_masks[fileNum]
            n_valid, n_total = chunk_stats[fileNum]

            with h5py.File(snapPath(basePath, snapNum, fileNum), 'r') as f:
                coords = f[gName]['Coordinates'][:]
                if fix_boundary_condition:
                    coords = periodic_boundary_condition_fix_coordinates(pos, coords, BoxSize)
                for field in fields:
                    if field == "Coordinates":
                        extract_2d(coords, idx, result[field], stored)
                    else:
                        load_chunk_data_fast(f, gName, field, idx, result[field], stored)

                stored += n_valid
                if verbose:
                    print(f"  Chunk {fileNum} were saved. Process {stored}/{total_valid}")

                del coords
        del cached_masks
        gc.collect()
        
    if load_halos == True or load_only_halos == True:
        if halo_fields is not None:
            if type(halo_fields) is str:
                halo_fields = [halo_fields]

            if "GroupPos" not in halo_fields:
                halo_fields.append("GroupPos")
        if verbose == True:
            print("  Extracing halos")
        all_halos = il.groupcat.loadHalos(basePath, snapNum, fields = halo_fields)
        GroupPos = all_halos["GroupPos"]
        mask = compute_mask(GroupPos, pos, rmax, geometry)
        halo_idx = np.where(mask == True)[0]
        if verbose == True:
            print(f"  {len(halo_idx)} halos were extracted from sub-volume!")
        if load_only_halos == True:
            return halo_idx
        else:
            return result, halo_idx
    else:
        return result
