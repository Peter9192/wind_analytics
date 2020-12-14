""" Three routines to identify low-level jets in wind profiles:

# 1. Simply loop over all profiles using np.apply_along_axis
llj_strength = wspd.reduce(detect_llj,dim='level')

# 2a. Fully vectorized function returning one llj characteristic at a time
# Most generic, can still be optimized by returning multiple outputs at once.

llj_strength = wspd.reduce(detect_llj_vectorized,dim='level',output='falloff')

# 2b. Call multiple times to combine relevant data ######
# Suboptimal, but xarray.DataArray.reduce only accepts single output arrays
get_height = lambda i: np.where(i>0,wspd.level.values[i],np.nan)
lljs = xr.merge([
    wspd.reduce(detect_llj_vectorized,dim='level').rename('falloff'),
    wspd.reduce(detect_llj_vectorized,dim='level',output='strength').rename('strength'),
    wspd.reduce(detect_llj_vectorized,dim='level',output='index').rename('height')
    ])
lljs.height.values = lljs.height.pipe(get_height)

# 3. High-level xarray implementation
# This is fastest for the my specific use case
lljs = detect_llj_xarray(wspd)

Peter Kalverla
March 2018
"""

import xarray as xr
import numpy as np

def detect_llj(x, axis=None, falloff=0, output='strength', inverse=False):
    """ Identify maxima in wind profiles.
        args:
        - x         : ndarray with wind profile data
        - axis      : specifies the vertical dimension
                      is internally used with np.apply_along_axis
        - falloff   : threshold for labeling as low-level jet
                      default 0; can be masked later, e.g. llj[falloff>2.0]
        - output    : specifiy return type: 'strength' or 'index'

        returns (depending on <output> argument):
        - strength  : 0 if no maximum identified, otherwise falloff strength
        - index     : nan if no maximum identified, otherwise index along
                      <axis>, to get the height of the jet etc.
    """

    def inner(x, output):
        if inverse:
            x = x[::-1, ...]

        # Identify local maxima
        x = x[~np.isnan(x)]
        dx = x[1:] - x[:-1]
        ind = np.where((np.hstack((dx, 0)) < 0) &
                       (np.hstack((0, dx)) >= 0))[0]

        # Last value of x cannot be llj
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]

        # Compute the falloff strength for each local maxima
        if ind.size: # this assumes height increases along axis!!!
            strength = np.array([x[i] - min(x[i:]) for i in ind])
            imax = np.argmax(strength)

        # Return jet_strength and index of maximum:
        if output=='strength':
            r = max(strength) if ind.size else 0
        elif output=='index':
            r = ind[imax] if ind.size else 0

        return r
    # Wrapper interface to apply 1d function to ndarray
    return np.apply_along_axis(inner,axis,x,output=output)


def detect_llj_vectorized(xs,axis=-1,output='falloff', mask_inv=False, inverse=False):
    """ Identify local maxima in wind profiles.
        args:
        - x         : ndarray with wind profile data
        - axis      : specifies the vertical dimension
        - output    : specifiy return type: 'falloff', 'strength' or 'index'
        - mask_inv  : use np.ma to mask nan values

        returns (depending on <output> argument and whether llj is identified):
        - falloff   : 0 or largest difference between local max and subseq min
        - strength  : 0 or wind speed at jet height
        - index     : -1 or index along <axis>
    """
    # Move <axis> to first dimension, to easily index and iterate over it.
    xv = np.rollaxis(xs,axis)

    if inverse:
        xv = xv[::-1, ...]

    if mask_inv:
        xv = np.ma.masked_invalid(xv)

    # Set initial arrays
    min_elem = xv[-1].copy()
    max_elem = np.zeros(min_elem.shape)
    max_diff = np.zeros(min_elem.shape)
    max_idx = np.ones(min_elem.shape,dtype=int) * (-1)

    # Start at end of array and search backwards for larger differences.
    for i, elem in reversed(list(enumerate(xv))):
        min_elem = np.minimum(elem, min_elem)
        new_max_identified = elem - min_elem > max_diff
        max_diff = np.where(new_max_identified, elem - min_elem, max_diff)
        max_elem = np.where(new_max_identified, elem, max_elem)
        max_idx = np.where(new_max_identified, i, max_idx)

    if output == 'falloff':
        r = max_diff
    elif output == 'strength':
        r = max_elem
    elif output == 'index':
        r = max_idx
    else:
        raise ValueError('Invalid argument for <output>: %s'%output)

    return r

def detect_llj_xarray(da, inverse=False):
    """ Identify local maxima in wind profiles.
        args:
        - da        : xarray.DataArray with wind profile data
        - inverse   : ecmwf stores the data upside-down

        returns:    : xarray.Dataset with vertical dimension removed containing:
        - falloff   : 0 or largest difference between local max and subseq min
        - strength  : 0 or wind speed at jet height
        - index     : -1 or index along <axis>

        Note: vertical dimension should be labeled 'level' and axis=1
    """
    # Move <axis> to first dimension, to easily index and iterate over it.
    xv = np.rollaxis(da.values,1)

    if inverse:
        xv = xv[::-1, ...]

    # Set initial arrays
    min_elem = xv[-1].copy()
    max_elem = np.zeros(min_elem.shape)
    max_diff = np.zeros(min_elem.shape)
    max_idx = np.ones(min_elem.shape,dtype=int) * (-1)

    # Start at end of array and search backwards for larger differences.
    for i, elem in reversed(list(enumerate(xv))):
        min_elem = np.minimum(elem, min_elem)
        new_max_identified = elem - min_elem > max_diff
        max_diff = np.where(new_max_identified, elem - min_elem, max_diff)
        max_elem = np.where(new_max_identified, elem, max_elem)
        max_idx = np.where(new_max_identified, i, max_idx)

    # Combine the results in a dataframe
    get_height = lambda i: np.where(i>0,da.level.values[i],da.level.values[-1])
    dims = da.isel(level=0).drop('level').dims
    coords = da.isel(level=0).drop('level').coords
    lljs = xr.Dataset({'falloff': (dims, max_diff),
                       'strength': (dims, max_elem),
                       'level': (dims, get_height(max_idx)),
                      }, coords = coords)
    print 'Beware! Level is also filled if no jet is detected!'
    print 'use ds.sel(level=lljs.level).where(lljs.falloff>0) to rid it'
'
    return lljs


if __name__=="__main__":
    ecmwf_levs = {
    110: 2081.09, 111: 1910.76, 112: 1750.63, 113: 1600.44,
    114: 1459.91, 115: 1328.70, 116: 1206.44, 117: 1092.73,
    118: 987.15, 119: 889.29, 120: 798.72, 121: 715.02,
    122: 637.76, 123: 566.54, 124: 500.95, 125: 440.61,
    126: 385.16, 127: 334.24, 128: 287.52, 129: 244.69,
    130: 205.44, 131: 169.51, 132: 136.62, 133: 106.54,
    134: 79.04, 135: 53.92, 136: 30.96, 137: 10.00
    }

    # Read ERA5 data; Reverse order of levels!
    era5 = xr.open_mfdataset('RAW/era5_*_ml.nc').sel(level=slice(None,124,-1))
    era5['level'] = [ecmwf_levs[x] for x in era5.level.values]

    # Compute wind speed
    wspd = (era5.u**2+era5.v**2)**.5

    # Detect low-level jets and write to output
    lljs = detect_llj_xarray(wspd)
    lljs.to_netcdf('lljs.nc')

    print 'ready'
