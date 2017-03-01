import numpy as np
import xarray as xray
import netCDF4
import warnings
import sys
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
import gsw

class POPFile(object):
    
    def __init__(self, fname,
                 hconst=None, pref=0., ah=-3e9, am=-2.7e10, is3d=False):
        """Wrapper for POP model netCDF files. 
            The units of diffusivity and viscosity are in [m^4/s]
        """
        self.nc = xray.open_dataset(fname, decode_times=False)
        self.Ny, self.Nx = self.nc.TAREA.shape  
        self._ah = ah
        self._am = am
        self.hconst = hconst
        
        ##########
        # mask
        ##########
        self.maskT = self.nc.KMT > 1
        self.maskU = self.nc.KMU > 1

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
            self.mask = self.nc.KMT > 1
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
            self._ahf[self.mask] = 0.   

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
            self.mask = self.nc.KMU > 1
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
                                   / uarea[j_eq, 0])**1.5, ~self.mask).filled(0.)
    
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
    
    def aggregate_latlon(self, lat, lon, varname='SST', istart=0, iend=3500, jstart=400, jend=2000,
                         roll=-1100, south=-90., north=90., west=-180., east=180., dlat=1, dlon=1):
        """Aggregate data based on lat-lon grids
        """
        T = self.nc[varname].roll(nlon=roll).values[:, jstart:jend, istart:iend]
        Nt = T.shape[0]
        if T.ndim == 3:
            Tagg = np.empty((Nt, int((north-south)/dlat), int((east-west)/dlon)))
        elif T.ndim == 4:
            Nt, Nz, Ny, Nx = T.shape
            Tagg = np.empty((Nt, Nz, int((north-south)/dlat), int((east-west)/dlon)))
        Tagg[:] = np.nan
        
        j = 0; s = south
        while s < north:
            i = 0; w = west
            while w < east:
                lonrange = np.array([w, w+dlon])
                latrange = np.array([s, s+dlat])
                lonmask = (lon >= lonrange[0]) & (lon < lonrange[1])
                latmask = (lat >= latrange[0]) & (lat < latrange[1])
                boxidx = lonmask & latmask # this won't necessarily be square
                irange = np.where(boxidx.sum(axis=0))[0]
                imin, imax = irange.min(), irange.max()
                jrange = np.where(boxidx.sum(axis=1))[0]
                jmin, jmax = jrange.min(), jrange.max()
                
                region_mask = self.maskT.values[jmin:jmax, imin:imax]
                if T.ndim == 3:
                    for t in range(Nt):
                        Tagg[t, j, i] = np.ma.mean(np.ma.masked_array(T[t, jmin:jmax, imin:imax], ~region_mask))
                
                w += dlon
                i += 1
            s += dlat
            j += 1
            
        return Tagg
