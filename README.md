# wind_analytics
Python functions for the analysis of wind data


## lljs.py

Three functions to identify low-level jets in wind profiles:

1. Simply loop over all profiles using `np.apply_along_axis`:

`llj_strength = wspd.reduce(detect_llj,dim='level')`

2. Fully vectorized function returning one llj characteristic at a time

Most generic, can still be optimized by returning multiple outputs at once.

`llj_strength = wspd.reduce(detect_llj_vectorized,dim='level',output='falloff')`

3. High-level xarray implementation

Fastest for my specific use case
`lljs = detect_llj_xarray(wspd)`
