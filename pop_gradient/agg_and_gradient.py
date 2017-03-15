import numpy as np
import xarray as xray
import pandas as pd
import netCDF4
import warnings
import sys
from warnings import warn
from scipy import linalg as lin
from scipy import interpolate as naiso
from scipy.spatial import cKDTree, KDTree
from itertools import repeat
import gsw

class POPFile(object):
    
    def __init__(self, fname,
                 hconst=None, pref=0., ah=-3e9, am=-2.7e10, is3d=False):
        """Wrapper for POP model netCDF files. 
            The units of diffusivity and viscosity are in [m^4/s]
        """
        self.nc = xray.open_dataset(fname, decode_times=False)
        # self.Ny, self.Nx = self.nc.TAREA.shape  
        # self._ah = ah
        # self._am = am
        # self.hconst = hconst
        
        ##########
        # mask
        ##########
        #self.maskT = self.nc.KMT > 1
        #self.maskU = self.nc.KMU > 1

        self.is3d = is3d
        if self.is3d:
            self.z_t = self.nc['z_t'][:]
            self.z_w_top = self.nc['z_w_top'][:]
            self.z_w_bot = self.nc['z_w_bop'][:]
            self.Nz = len(self.z_t)
            kmt = nc['KMT'][:]
            self.maskT3d = xray.DataArray(np.zeros((self.Nz, self.Ny, self.Nx), dtype='b'), coords=kmt.coords, dims=kmt.dims)
            Nz = maskT3d.shape[0]
            for k in range(Nz):
                self.maskT3d[k] = (kmt<=k)
          
        
    def initialize_gradient_operator(self, field='tracer'):
        """Needs to be called before calculating gradients
        """
        tarea = self.nc.TAREA.values
        self.tarea = tarea
        tarea_r = 1e4 * np.ma.masked_invalid(tarea**-1).filled(0.)
        self.tarea_r = tarea_r
        
        dxt = 1e-2 * self.nc.DXT.values
        self.dxt = dxt
        # dxt_r = 1e2 * self.nc.DXT.values**-1
        self.dxt_r = dxt**-1
        dyt = 1e-2 * self.nc.DYT.values
        self.dyt = dyt
        # dyt_r = 1e2 * self.nc.DYT.values**-1
        self.dyt_r = dyt**-1
        dxu = 1e-2 * self.nc.DXU.values
        self.dxu = dxu
        # dxu_r = 1e2 * self.nc.DXU.values**-1
        self.dxu_r = dxu**-1
        dyu = 1e-2 * self.nc.DYU.values
        self.dyu = dyu
        # dyu_r = 1e2 * self.nc.DYU.values**-1
        self.dyu_r = dyu**-1
        
        ############
        # Tracer
        ############
        if field == 'tracer':
            # self.mask = self.nc.KMT > 1
            ###########
            # raw grid geometry
            ###########
            work1 = (self.nc['HTN'][:] / self.nc['HUW'][:]).values
            dtn = work1*tarea_r
            dts = np.roll(work1,-1,axis=0)*tarea_r

            work1 = (self.nc['HTE'][:] / self.nc['HUS'][:]).values
            dte = work1*tarea_r
            dtw = np.roll(work1,-1,axis=1)*tarea_r
            
            ############
            # boundary conditions
            ############
            kmt = self.nc['KMT'].values > 1
            kmtn = np.roll(kmt,-1,axis=0)
            kmts = np.roll(kmt,1,axis=0)
            kmte = np.roll(kmt,-1,axis=1)
            kmtw = np.roll(kmt,1,axis=1)

            self._cn = np.where( kmt & kmtn, dtn, 0.)
            self._cs = np.where( kmt & kmts, dts, 0.)
            self._ce = np.where( kmt & kmte, dte, 0.)
            self._cw = np.where( kmt & kmtw, dtw, 0.)
            
            #############
            # mixing coefficients
            #############
            j_eq = np.argmin(self.nc['ULAT'][:,0].values**2)
            self._ahf = (tarea / self.nc['UAREA'].values[j_eq,0])**1.5
            self._ahf[self.maskT] = 0.   

            ###########
            # stuff for gradient
            # reciprocal of dx and dy (in meters)
            ###########
            self._kmaske = np.where(kmt & kmte, 1., 0.)
            self._kmaskn = np.where(kmt & kmtn, 1., 0.) 
            
        ############
        # Momentum
        ############
        elif field == 'momentum':
            p5 = .5
            c2 = 2.
            # self.mask = self.nc.KMU > 1
            hus = 1e-2 * self.nc.HUS.values
            self._hus = hus
            hte = 1e-2 * self.nc.HTE.values
            self._hte = hte
            huw = 1e-2 * self.nc.HUW.values
            self._huw = huw
            htn = 1e-2 * self.nc.HTN.values
            self._htn = htn

            uarea = self.nc.UAREA.values
            self.uarea = uarea
            uarea_r = 1e4 * np.ma.masked_invalid(uarea**-1).filled(0.)
            self.uarea_r = uarea_r

            ###########
            # coefficients for \nabla**2(U) (without metric terms)
            ###########
            work1 = hus * hte**-1
            dus = work1 * uarea_r
            self._dus = dus
            dun = np.roll(work1, 1, axis=0) * uarea_r
            self._dun = dun
            work1 = huw * htn**-1
            duw = work1 * uarea_r
            self._duw = duw
            due = np.roll(work1, 1, axis=1) * uarea_r
            self._due = due

            ###########
            # coefficients for metric terms in \nabla**2(U, V)
            ###########
            kxu = (np.roll(huw, 1, axis=1) - huw) * uarea_r
            kyu = (np.roll(hus, 1, axis=0) - hus) * uarea_r

            #East-West
            work1 = (hte - np.roll(hte, -1, axis=1)) * tarea_r
            work2 = np.roll(work1, 1, axis=1) - work1
            dxkx = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxu_r
            self._dxkx = dxkx
            work2 = np.roll(work1, 1, axis=0) - work1
            dykx = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyu_r
            self._dykx = dykx

            # North-South
            work1 = (htn - np.roll(htn, -1, axis=0)) * tarea_r
            work2 = np.roll(work1, 1, axis=0) - work1
            dyky = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyu_r
            self._dyky = dyky
            work2 = np.roll(work1, 1, axis=1) - work1
            dxky = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxu_r
            self._dxky = dxky

            dum = - (dxkx + dyky + c2*(kxu**2 + kyu**2))
            self._dum = dum
            dmc = dxky - dykx
            self._dmc = dmc

            ###########      
            # coefficients for metric mixing terms which mix U,V.
            ###########
            dme = c2*kyu * (htn + np.roll(htn, 1, axis=1))**-1
            self._dme = dme
            dmn = -c2*kxu * (hte + np.roll(hte, 1, axis=0))**-1
            self._dmn = dmn

            duc = - (dun + dus + due + duw)
            self._duc = duc
            dmw = -dme
            self._dmw = dmw
            dms = -dmn
            self._dms = dms
            
            #############
            # mixing coefficients
            #############
            j_eq = np.argmin(self.nc['ULAT'][:,0].values**2)
            self._amf = np.ma.masked_array((uarea 
                                   / uarea[j_eq, 0])**1.5, ~self.maskU).filled(0.)
    
    def interpolate_2d(self, Ti, meth='linear'):
        """Interpolate a 2D field
        """
        Ny, Nx = Ti.shape
        x = np.arange(0, Nx)
        y = np.arange(0, Ny)
        X, Y = np.meshgrid(x, y)
        Zr = Ti.ravel()
        Xr = np.ma.masked_array(X.ravel(), Zr.mask)
        Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
        Xm = np.ma.masked_array(Xr.data, ~Xr.mask).compressed()
        Ym = np.ma.masked_array(Yr.data, ~Yr.mask).compressed()
        Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                        Zr.compressed(), np.array([Xm,Ym]).T, method=meth)
        Znew = Zr.data
        Znew[Zr.mask] = Zm
        Znew.shape = Ti.shape

        return Znew
    
    def _tracer_gradient(self, varname):
        T = self.nc[varname].values
        dTdx = ((T - np.roll(T, 1, axis=2)) / (np.roll(self.dxu, 1, axis=1) + np.roll(np.roll(self.dxu, 1, axis=1), 1, axis=0)) 
                    + (np.roll(T, -1, axis=2) - T) / (self.dxu + np.roll(self.dxu, 1, axis=0)))
        dTdy = ((T - np.roll(T, 1, axis=1)) / (np.roll(self.dyu, 1, axis=0) + np.roll(np.roll(self.dyu, 1, axis=0), 1, axis=1)) 
                    + (np.roll(T, -1, axis=1) - T) / (self.dyu + np.roll(self.dyu, 1, axis=1)))
        return dTdx, dTdy
    
    def gradient(self, varname='SST', field='tracer'):
        #######
        # metric terms
        #######
        # kxu = ((np.roll(self._huw, 1, axis=1) - 
        #                          self._huw)) * self.uarea_r
        # kyu = ((np.roll(self._hus, 1, axis=0) - 
        #                          self._hus)) * self.uarea_r
        if field == 'tracer':
            return self._tracer_gradient(varname)
        
    def aggregate_grid(self, varname='SST', factor=2, mean=True):
        """ Aggregates data based on grid points defined by factor
        """
        data = self.nc[varname].values
        ndim = data.ndim
        shape = data.shape
        # promote single value to list
        if isinstance(factor, int):
            factors = ndim * [factor,]
        # print 'ndim: ', ndim, ' factors: ', factors
        # check we have the right number of dimensions
        assert len(factors) == ndim
        # make sure shapes are compatible
        for s, fac in zip(shape, factors):
        # print 's: ', s, ' fac: ', fac
            assert s % factor == 0
        out = 0
        # it is lazy to use a set...don't have to figure out the necessary logic
        slices = []
        for start_indices in product(*[range(f) for f in factors]):
            slices.append(
                [slice(sidx, s, factor) for sidx, s in zip(start_indices, shape)]
            )

        # how would we generalize to other reduce functions?
        result = reduce(np.add, [data[sl] for sl in slices])
        if mean:
            result /= len(slices)
        return result
    
    def aggregate_latlon(self, lat, lon, newlat, newlon, mask,
                         varname, istart, iend, jstart, jend, 
                         nroll, cython, *args):
        """
        Parameters
        --------------
        ds : xarray.Dataset
            raw data
        lat : numpy.array
            raw latitude
        lon: numpy.array
            raw longitude
        newlat : numpy.array
            latitude coordinate to regrid on
        newlon : numpy.array
            longitude coordinate to regrid on
        mask : xarray.Dataset
            mask for land
        varname : string
            name of tracer
        istart, iend, jstart, jend : integer
            indexes of zonal and meridional extent of the raw data
        nroll : integer
            number of indicies to roll the data zonally
        cython : boolean
            criterion whether to use KDTree or cKDTree
        *args : list
            coordinate names

        Returns
        -------------
        da : xarray.Dataset
            new dataset with labels added
        """
        data = self.nc[varname].where(mask).roll(nlon=nroll)
        assert data.values.ndim == 3
        assert lat.ndim == 2
        assert lon.ndim == 2
        assert np.isnan(lat).any() == False
        assert np.isnan(lon).any() == False
        
        ncoords = len(data.coords.keys())
        assert ncoords == len(args)
        # print ncoords
        timecoords = args[0]
        latcoords = args[1]
        loncoords = args[2]
        if len(args) > 3:
            unnescoords = []
            for i in range(3, ncoords):
                unnescoords.append(args[i])
            T = data.reset_coords(names=unnescoords, 
                                        drop=True).copy()[:, jstart:jend, istart:iend]

        T_nlon = xray.DataArray(range(T.shape[2]), 
                                                       dims=[loncoords],
                                                       coords={loncoords: range(T.shape[2])})
        T_nlat = xray.DataArray(range(T.shape[1]), 
                                                       dims=[latcoords],
                                                       coords={latcoords: range(T.shape[1])})
        T.coords[loncoords] = T_nlon
        T.coords[latcoords] = T_nlat
        
        original_coords = zip(lat.ravel(), lon.ravel())
        newyx = zip(newlat.ravel(), newlon.ravel())
        if cython:
            tree = cKDTree(newyx)
        else:
            tree = KDTree(newyx)
        distance, index = tree.query(original_coords)
        
        newlabel = np.empty(len(index), dtype=tuple)
        index_test = np.zeros_like(index)
        for i in range(len(index)):
            index_test[i] = index[i]
            newlabel[i] = newyx[index[i]]
        
        t_coords = T[0].stack(points=(latcoords, 
                                      loncoords)).reset_coords(names=timecoords, drop=True)['points']
        da_index = xray.DataArray(index, 
                                                       dims=t_coords.dims,
                                                       coords=t_coords.coords)
        
        da_label = xray.DataArray(newlabel, 
                                                       dims=t_coords.dims,
                                                       coords=t_coords.coords)
        
        
        Nt = T.shape[0]
        da_numpy = np.zeros((Nt, len(newlat[:, 0]), len(newlon[0, :])))
        for t in range(Nt):
            Ti = T[t].stack(points=(latcoords, 
                                   loncoords)).reset_coords(names=timecoords, drop=True).copy()
            Ti.coords['index'] = da_index
            Ti.coords['label'] = da_label
            Ti_grouped = Ti.groupby('label').mean().to_dataset(name=varname)

            arrays = [[i for item in newlat[:, 0] for i in repeat(item, len(newlon[0, :]))], 
                          np.tile(newlon[0, :], len(newlat[:, 0]))]
            tuples = list(zip(*arrays))
            Ti_panda_index = pd.MultiIndex.from_tuples(tuples, names=['lat', 'lon'])
            Ti_panda = pd.Series(Ti_grouped[varname], index=Ti_panda_index)
            Ti_panda_unstacked = Ti_panda.unstack(level=-1)
            da_numpy[t] = Ti_panda_unstacked.values

        da = xray.DataArray(da_numpy, dims=['day', 'lat', 'lon'], 
                                              coords={'day': range(Nt), 'lat': newlat[:, 0], 'lon': newlon[0, :]}).to_dataset(name=varname)

        return da
