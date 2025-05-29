### api_ATLAS.py
# Yukei Murakami @ JHU, 2024
# this is meant for my personal use, so it may contain unnecessary or deprecated functions.

# This module provides tools to query ATLAS forced photometry data and analyze it.
# The class QueryATLAS is used to query the ATLAS forced photometry data.
# The class AtlasPhotometry is used to quickly import the photometry data file and analyze the data.

#### example for querying data ####
# task_url = api.request_phot(radeg,decdeg,mjd_min,mjd_max,use_reduced=False) # request photometry
# api.save_phot(task_url,folder='phot_output') # save the photometry data in the phot_output folder

#### example for analyzing data ####
# phot = AtlasPhotometry('phot_file.txt',objname='object_name')
# phot.cut(fluxerr_max=1000) # apply cuts. I always cut data at fluxerr_max=3000 at least.
# phot.plot_lc() # plot the light curve in SNANA flux scale
# phot.fit_lc(preset_vals={'z':0.05678}) # run SNcosmo. see note below
# phot.to_SNANA('output_file.txt') # save the photometry in SNANA data format
#
# phot.to_SNANA can also take a header argument to add metadata to the output file.
# headers = {
#     'SURVEY': 'ATLAS',
#     'SNID': objname,
#     'RA': f'{ra:.5f}',
#     'DECL': f'{dec:.5f}',
#     'RA_HOST': f'{ra_host:.5f}',
#     'DEC_HOST': f'{dec_host:.5f}',
#     'HOSTGAL_LOGMASS': f'{logmass:.2f} +- {logmass_err:.2f}',
#     'REDSHIFT_HELIO': f'{redshift:.6f} +- 0.0001',
#     'FILTERS': 'co'
# }
# phot.to_SNANA('output_file.txt',header=headers)

#### NOTE: light curve fitting requires some setup.
# 1. Download the filter data from SVO database and run the following:
# import sncosmo
# cyan_wav,cyan_filt = np.loadtxt('filters/ATLAS_cyan.dat',unpack=True)
# orange_wav,orange_filt = np.loadtxt('filters/ATLAS_orange.dat',unpack=True)
# sncosmo.register(sncosmo.Bandpass(orange_wav, orange_filt, name='o'))
# sncosmo.register(sncosmo.Bandpass(cyan_wav, cyan_filt, name='c'))
#
# 2. Select the model to fit. The default is 'salt2'. 
# You can change it by running a line like this:
# phot._prep_sncosmo_model(source='hsiao',params_to_fit=['z', 't0', 'amplitude'])
# if sources other than 'salt2' are used, the bounds should be adjusted.
# phot.plot_lc(bounds=dict_of_tuples)

# 3. set preset (fixed) values. For example, if the redshift is known:
# phot.fit_lc(preset_vals={'z':0.123})

import json
import requests
from tqdm.auto import tqdm
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clip, sigma_clipped_stats



class QueryATLAS():
    def __init__(self,url='https://fallingstar-data.com/forcedphot',
                 api_token=None):
        self.url = url
        self.api_token = api_token
        self.headers = self.create_headers()

    def create_headers(self,):
        headers = {'Authorization': f'Token {self.api_token}', 'Accept': 'application/json'}
        return headers
            
    def handle_throttle(self, response, waittime_default=10):
        ''' predict the waittime based on throttling message '''
        message = resp.json()["detail"]
        print(f'{resp.status_code} {message}')
        t_sec = re.findall(r'available in (\d+) seconds', message)
        t_min = re.findall(r'available in (\d+) minutes', message)
        if t_sec:
            waittime = int(t_sec[0])
        elif t_min:
            waittime = int(t_min[0]) * 60
        else:
            waittime = waittime_default
            
    def wait_if_throttled(self, response, waittime_default=10):
        ''' wait while the server is throttling '''
        if response.status_code != 429:
            return False
        waittime = self.handle_throttle(response, waittime_default)
        print(f'Waiting {waittime} seconds')
        time.sleep(waittime)
        return True
                    
    def request_phot(self, maxiter=10, **kwargs):
        ''' request photometry to the server '''
        task_url = None
        for _ in range(maxiter):
            with requests.Session() as s:
                resp = s.post(f'{self.url}/queue/', headers=self.headers, data=kwargs)
                if self.wait_if_throttled(resp):
                    continue
                elif resp.status_code == 201:  # successfully queued
                    task_url = resp.json()['url']
                    return task_url
                else:
                    raise ValueError(f'ERROR {resp.status_code} {resp.json()}')
        raise ValueError('Maxiter reached. Task not queued. This is likely because the server is busy.')

    def request_images(self, task_url, maxiter=10, **kwargs):
        ''' request differential images to the server '''
        for _ in range(maxiter):
            with requests.Session() as s:
                resp = s.get(task_url + 'requestimages', headers=self.headers)
                if self.wait_if_throttled(resp):
                    continue
                elif resp.status_code == 200:  # successfully queued
                    result = resp.json()['results'][0]
                    if 'result_imagezip_url' in result:
                        return result['result_imagezip_url']
                    else:
                        return None # no images available yet
                else:
                    raise ValueError(f'ERROR {resp.status_code} {resp.json()}')
        raise ValueError('Maxiter reached. Task not queued. This is likely because the server is busy.')

    def get_task_status(self, task_url, maxiter=10):
        for _ in range(maxiter):
            with requests.Session() as s:
                resp = s.get(task_url, headers=self.headers)
                if self.wait_if_throttled(resp):
                    continue
                elif resp.status_code == 200:  # successfully queued
                    return resp.json()
                else:
                    raise ValueError(f'ERROR {resp.status_code} {resp.json()}')
        raise ValueError('Maxiter reached. Task not queued. This is likely because the server is busy.') 
    
    def get_data_urls(self, task_url, maxiter=10):
        ''' request differential images to the server '''
        result = self.get_task_status(task_url, maxiter=maxiter)
        return_dict = {}
        return_dict['result_imagezip_url'] = result.get('result_imagezip_url',None)
        return_dict['result_url'] = result.get('result_url',None)
        return return_dict
 
    def get_phot_text(self, task_url, maxiter=10):
        phot_url = self.get_data_urls(task_url)['result_url']
        if phot_url is None:
            raise ValueError('No photometry available yet')
        for _ in range(maxiter):
            with requests.Session() as s:
                resp = s.get(phot_url, headers=self.headers)
                if self.wait_if_throttled(resp):
                    continue
                elif resp.status_code == 200:
                    textdata = resp.text
                    return textdata, phot_url
                else:
                    raise ValueError(f'ERROR {resp.status_code} {resp.json()}')
        raise ValueError('Maxiter reached. Task not queued. This is likely because the server is busy.') 

    def save_phot(self, task_url, folder='./', maxiter=10):
        text,phot_url = self.get_phot_text(task_url, maxiter=maxiter)
        filename = os.path.join(folder, os.path.split(phot_url)[-1])
        if os.path.exists(filename):
            return filename
        with open(filename,'w') as f:
            f.write(text.replace('###',''))
        return filename

    def save_images(self, task_url, path='./diff_images/', 
                    maxiter=10, block_size = 1024):
        ''' download and unzip images into a folder '''
        taskid = os.path.split(task_url)[1]
        
        img_url = self.get_data_urls(task_url)['result_imagezip_url']
        if img_url is None:
            raise ValueError('Images are not ready yet')
        
        local_zip_path = path+'.zip'
        
        # download zip
        with requests.get(img_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_zip_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=block_size),
                                  desc='Downloading zip file',
                                  total=int(total_size/block_size),
                                  unit = 'KB',
                                  leave=False):
                    f.write(chunk)
        
        # Step 2: Unzip the file
        unzip_loc = os.path.splitext(local_zip_path)[0]
        os.makedirs(unzip_loc, exist_ok=True)
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_loc)
        
        # Remove the zip file after extraction
        os.remove(local_zip_path)
        
        return unzip_loc
    
class AtlasPhotometry():
    def __init__(self,phot_file,objname='',zp=27.5,
                 warn_template_changes=True):
        self.phot_file = phot_file
        self.objname = objname
        self.zp = zp

        # read photometry
        self.df_phot = pd.read_csv(phot_file,
                                   sep='\s+',#delim_whitespace=True,
                                   escapechar='#')
        self.df_phot.columns=self.df_phot.columns.str.replace('#','') # some files have '###' in the header
        self._mag = self.df_phot['m'].values
        self._magerr = self.df_phot['dm'].values
        self._mjd = self.df_phot['MJD'].values
        self._filters = self.df_phot['F'].values
        self._flux_ATLAS = self.df_phot['uJy'].values
        self._fluxerr_ATLAS = self.df_phot['duJy'].values
        self._chi2dof = self.df_phot['chi/N'].values
        self.cuts = np.ones(len(self._mag),dtype=bool)
        self.parse_telescope_info()
        self.calc_flux()
        self.cut()
        self.warn_template_changes = warn_template_changes
        self.check_template_change(warn_template_changes)
        
    def check_template_change(self,warn_template_changes=True):
        ''' check if the template changes during the light curve '''
        t12 = 58417 # MJD for template change 1->2
        t23 = 58882 # MJD for template change 2->3
        
        self.t12 = t12
        self.t23 = t23
        self.template_change_12 = False
        self.template_change_23 = False
        if self.mjd.min() < t12 and self.mjd.max() > t12:
            if warn_template_changes:
                print(f'!!! WARNING: Data MJD range contains Template change 1 -> 2 (MJD={t12}) !!!')
            self.template_change_12 = True
        if self.mjd.min() < t23 and self.mjd.max() > t23:
            if warn_template_changes:
                print(f'!!! WARNING: Data MJD range contains Template change 2 -> 3 (MJD={t23}) !!!')
            self.template_change_23 = True
        
    def parse_telescope_info(self):
        import importlib.resources as ir
        filename_cyan = ir.files(__package__).joinpath("chip_info.dat")
        df_chip = pd.read_csv(filename_cyan,sep='\s+')
        df_chip = df_chip.replace('current',999999)
        self._sitecam = self.df_phot['Obs'].str.slice(0, 3).values
        
        self._chipid = np.ones_like(self._sitecam,dtype=float)
        for i,(sitecam,mjd) in enumerate(zip(self._sitecam,self._mjd)):
            idx = df_chip.index.values[df_chip['sitecam'].eq(sitecam) & df_chip['mjd_min'].astype(float).le(mjd) & df_chip['mjd_max'].astype(float).ge(mjd)]
            if len(idx) == 0:
                idx = [np.nan]
            self._chipid[i] = idx[0]
        self.df_chip = df_chip
                
    def _format_df_phot(self):
        ''' specific to ATLAS dataset.'''
        self.df_phot['MJD'] = self._mjd
        self.df_phot['FLT'] = self._filters
        self.df_phot['FIELD'] = -1
        self.df_phot['FLUXCAL'] = self._flux
        self.df_phot['FLUXCALERR'] = self._flux_err
        
    def cut_data_at_template_change(self,pkmjd,retain_phase=[-20,50]):
        ''' cut the data before/after template change if the template change occurs outside the retain_phase range'''
        if self.template_change_12:
            cuts = self.cuts.copy()
            if self.t12 < pkmjd-retain_phase[0]:
                self.cut(s=cuts,mjd_min=self.t12)
            elif self.t12 > pkmjd+retain_phase[1]:
                self.cut(s=cuts,mjd_max=self.t12)
        if self.template_change_23:
            cuts = self.cuts.copy()
            if self.t23 < pkmjd-retain_phase[0]:
                self.cut(s=cuts,mjd_min=self.t23)
            elif self.t23 > pkmjd+retain_phase[1]:
                self.cut(s=cuts,mjd_max=self.t23) 
        
    def reapply_cuts(self):
        s = self.cuts
        if hasattr(self,'_mag'):
            self.mag = self._mag[s]
            self.magerr = self._magerr[s]
        self.mjd = self._mjd[s]
        self.filters = self._filters[s]
        self.cuts = s        
        self.calc_flux()

    def cut(self,mag_min=None,mag_max=None,
            magerr_min=None,magerr_max=None,
            mjd_min=None,mjd_max=None,
            sigma_max=None,res_max=None,
            flux_min=None,flux_max=None,
            fluxerr_min=None,fluxerr_max=None,
            sncosmo_res_max=None,
            chi2dof_max=None,
            flux_pull_max=None,
            filters=['o','c'],
            chipid=None,
            s = None,
            **kwargs):
        if s is None:
            s = np.isfinite(self._flux)
        else:
            s = s.copy()
        s &= np.isfinite(self._flux_err)
        # s = np.ones(len(self._mjd),dtype=bool)
        if mag_min is not None:
            s &= self._mag >= mag_min
        if mag_max is not None:
            s &= self._mag <= mag_max
        if magerr_min is not None:
            s &= self._magerr >= magerr_min
        if magerr_max is not None:
            s &= self._magerr <= magerr_max
        if mjd_min is not None:
            s &= self._mjd >= mjd_min
        if mjd_max is not None:
            s &= self._mjd <= mjd_max
        if flux_min is not None:
            s &= self._flux >= flux_min
        if flux_max is not None:
            s &= self._flux <= flux_max
        if fluxerr_min is not None:
            s &= self._flux_err >= fluxerr_min
        if fluxerr_max is not None:
            s &= self._flux_err <= fluxerr_max
        if chi2dof_max is not None:
            s &= self._chi2dof <= chi2dof_max
        if chipid is not None:
            s &= self._chipid == chipid
        # s &= self.df_phot['F'].isin(filters)
        s &= np.isin(self._filters,filters)
    
            
        if sigma_max is not None:
            if not hasattr(self,'_res_sigma'):
                self._get_residual()
            # save the objects that passed everything else except this
            s_res_sigma = self._res_sigma > sigma_max
            s_clipped = s.copy() & s_res_sigma
            s &= ~s_res_sigma
            self.s_clipped = s_clipped
        
        if res_max is not None:
            if not hasattr(self,'_res'):
                self._get_residual()
            # save the objects that passed everything else except this
            s_res = np.abs(self._res) > res_max
            if hasattr(self,'s_clipped'):
                s_clipped = self.s_clipped | (s.copy() & s_res)
            else:
                s_clipped = s.copy() & s_res
            s &= ~s_res
            self.s_clipped = s_clipped

        if sncosmo_res_max is not None:
            # save the objects that passed everything else except this
            s_sncosmo_res = np.abs(self._sncosmo_residual_flux) > sncosmo_res_max
            if hasattr(self,'s_clipped'):
                s_clipped = self.s_clipped | (s.copy() & s_sncosmo_res)
            else:
                s_clipped = s.copy() & s_sncosmo_res
            s &= ~s_sncosmo_res
            self.s_clipped = s_clipped

        if flux_pull_max is not None:
            if not hasattr(self,'_flux_pull'):
                print('Flux pull has not been calculated. Run calc_nightly_flux_pull() first.')
            else:
                s &= np.abs(self._flux_pull) <= flux_pull_max
                s &= self._pull_clip

        if hasattr(self,'_mag'):
            self.mag = self._mag[s]
            self.magerr = self._magerr[s]
        if hasattr(self,'_flux_pull'):
            self.flux_pull = self._flux_pull[s]
        self.mjd = self._mjd[s]
        self.filters = self._filters[s]
        if hasattr(self,'_chi2dof'):
            self.chi2dof = self._chi2dof[s]
        if hasattr(self,'_chipid'):
            self.chipid = self._chipid[s]
        self.cuts = s        
        self.calc_flux()
        
    def plot_lc(self,plot_interp=False,ax=None):
        if not hasattr(self,'mjd'):
            self.cut()
        
        ymin,ymax = self.flux.min(),self.flux.max()
        # plot & save
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(8,6))
        else:
            fig = ax.figure
        for filt,filtername in zip(['o','c'],['orange','cyan']):
            s = self.filters == filt
            ax.errorbar(self.mjd[s],self.flux[s],self.flux_err[s],
                        fmt='o',ms=3,label=filtername,color=filtername)
            
            if hasattr(self,'s_clipped'):
                _s = self.s_clipped & (self._filters == filt)
                ax.errorbar(self._mjd[_s],self._flux[_s],self._flux_err[_s],
                            fmt='o',ms=3,color=filtername)
                ax.scatter(self._mjd[_s],self._flux[_s],
                           marker='x',color='red',s=50,zorder=20)
                # if _s.sum() > 0:
                #     ymax = max(self._flux[_s].max(),ymax)
                #     ymin = min(self._flux[_s].min(),ymin)
        
        if plot_interp:
            from csaps import csaps
            interp_func = csaps(self.mjd,self.flux,
                                normalizedsmooth=True,
                                smooth=self.csaps_smooth)
            x_th = np.linspace(self.mjd.min(),self.mjd.max(),1000)
            y_th = interp_func(x_th)
            plt.plot(x_th,y_th,c='yellowgreen',lw=3)
            
        margin = (ymax-ymin)*0.1
        ax.set_ylim(ymin-margin,ymax+margin)
        ax.tick_params(direction='in')    
        ax.set_xlabel('MJD',fontsize=13)
        ax.set_ylabel('SNANA flux',fontsize=13)
        ax.set_title(self.objname,fontsize=13,x=0.3)
        ax.legend(frameon=False,bbox_to_anchor=[0.6,1.0],loc='lower left',ncols=2)
        ax.axhline(0,c='k',ls=':',lw=1,zorder=0)
        return fig,ax
     
    def calc_flux(self):
        if not hasattr(self,'_flux'):
            _flux = self._flux_ATLAS * 10**(-0.4*(23.9-self.zp))
            _flux_err = self._fluxerr_ATLAS * 10**(-0.4*(23.9-self.zp))
            
            # calculate SNR -- applying masks to avoid divide by zero warnings
            _flux_snr = np.zeros_like(_flux)
            finite_err = np.isfinite(_flux_err) & (_flux_err > 0)
            _flux_snr[finite_err] = np.abs(_flux[finite_err] / _flux_err[finite_err])
            
            self._flux = _flux
            self._flux_err = _flux_err
            self._flux_snr = _flux_snr
        self.flux = self._flux[self.cuts]
        self.flux_err = self._flux_err[self.cuts]
        if hasattr(self,'_flux_snr'):
            self.flux_snr = self._flux_snr[self.cuts]
        
    def subtract_flux_offset(self,flux_offset_c=0.0,flux_offset_o=0.0,
                                flux_offset_c_err=0.0,flux_offset_o_err=0.0):
        ''' subtract flux offset from the fluxes.'''
        if not hasattr(self,'_flux'):
            self.calc_flux()
            
        if flux_offset_c != 0.0:
            self._flux[self._filters=='c'] -= flux_offset_c
        if flux_offset_o != 0.0:
            self._flux[self._filters=='o'] -= flux_offset_o
        if flux_offset_c_err != 0.0:
            _err = self._flux_err[self._filters=='c']
            self._flux_err[self._filters=='c'] = np.sqrt(_err**2 + flux_offset_c_err**2)
        if flux_offset_o_err != 0.0:
            _err = self._flux_err[self._filters=='o']
            self._flux_err[self._filters=='o'] = np.sqrt(_err**2 + flux_offset_o_err**2)
        self.calc_flux()  # recalculate flux and flux_err
        
    def to_SNANA(self,outfile='',header={'SURVEY':'ATLAS'},
                 columns_to_output=['MJD','FLT', 'FIELD', 'FLUXCAL', 'FLUXCALERR']):
        self._format_df_phot()
        
            # output file        
        with open(outfile,'w') as g:
            # write header
            
            for key,val in header.items():
                g.write(f'{key}: {val}\n')

            g.write('\n')
            g.write('# ========================================\n')
            
            # more header
            varlist = columns_to_output
            g.write(f'NOBS: {len(self.mjd)}\n')
            g.write(f'NVAR: {len(varlist)}\n')
            varlist_str = ' '.join(varlist)
            g.write(f'VARLIST: {varlist_str}\n')

            for i in self.df_phot.index.values[self.cuts]:
                values = self.df_phot.loc[i,columns_to_output].values.astype(str)
                g.write('OBS:   '+' '.join(values)+'\n')
                # mjd = self.mjd[i]
                # filt = self.filters[i]
                # flux = self.flux[i]
                # flux_err = self.flux_err[i]                
                # g.write(f'OBS:   {mjd} {filt} -1 {flux:.3f} {flux_err:.3f}\n')
                
    def _determine_best_csaps_smoothing(self,n_splits=5):
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        from csaps import csaps
        from time import time
        
        csaps_smooth_values = np.linspace(0.1,0.999,200)
        best_csaps_smooth = None
        best_score = float('inf')

        for csaps_smooth in csaps_smooth_values:
            scores = []
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int((time()%0.01)*1e6))
            for train_index, test_index in kf.split(self.mjd):
                time_train, time_test = self.mjd[train_index], self.mjd[test_index]
                mag_train, mag_test = self.mag[train_index], self.mag[test_index]

                interp_func = csaps(time_train,mag_train,normalizedsmooth=True,smooth=csaps_smooth)
                mag_fit_test = interp_func(time_test)
                
                # Calculate the RMSE for the test set
                rmse = np.sqrt(mean_squared_error(mag_test, mag_fit_test))
                scores.append(rmse)

            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                best_csaps_smooth = csaps_smooth
                
            plt.scatter(csaps_smooth,avg_score)
        self._best_csaps_smooth = best_csaps_smooth
        return best_csaps_smooth
    
    def _get_residual(self,csaps_smooth=None,plot=False,**kwargs):
        
        if csaps_smooth is None:
            if not hasattr(self,'_best_csaps_smooth'):
                self._determine_best_csaps_smoothing()
            csaps_smooth = self._best_csaps_smooth
        from csaps import csaps
        
        # interpolate the data
        # print('using csaps smoothing:',csaps_smooth)
        interp_func = csaps(self.mjd,self.mag,
                            normalizedsmooth=True,
                            smooth=csaps_smooth)
        res = self.mag - interp_func(self.mjd)
        _res = self._mag - interp_func(self._mjd) # all
        res_sigma = np.abs(res) / self.magerr
        _res_sigma = np.abs(_res) / self._magerr # all
        x_th = self.mjd
        y_th = interp_func(x_th)
        
        if plot:
            # plot
            fig,ax = plt.subplots(1,1,figsize=(8,6))
            plt.subplots_adjust(left=0.1,right=0.9,bottom=0.2,top=0.9)
            ax2 = fig.add_axes([0.1,0,0.8,0.2])
            
            ax.errorbar(self.mjd,self.mag,self.magerr,fmt='o',c='k',ms=2)
            ax.plot(x_th,y_th,c='yellowgreen',lw=3)
            ax2.axhline(0,c='k',ls='--')
            ax2.errorbar(self.mjd,res,self.magerr,fmt='o',c='k',ms=2)
            ax2.set_xlim(ax.get_xlim())
            
            ax.invert_yaxis()
            ax2.invert_yaxis()
        self._res = _res
        self._res_sigma = _res_sigma
        self.csaps_smooth = csaps_smooth
        
    def clip_outliers(self,sigma=15,res_max=2,maxiter=10,csaps_smooth=None,**kwargs):
        ''' iteratively clip sigma. kwargs are passed to self.cut() '''
      
        self._get_residual(csaps_smooth=csaps_smooth,**kwargs)  
        for _ in range(maxiter):
            self.cut(sigma_max=sigma,res_max=res_max,**kwargs)
            self._get_residual(csaps_smooth=csaps_smooth,**kwargs)
            
    def _make_sncosmo_data(self):
        table = Table()
        table['mjd'] = self.mjd
        table['filter'] = self.filters
        table['flux'] = self.flux
        table['flux_err'] = self.flux_err
        table['zp'] = self.zp
        table['zpsys'] = 'ab'
        
        self.sncosmo_data = table
        return table

    def _prep_sncosmo_model(self,source='salt2',
                            params_to_fit=['z', 't0', 'x0', 'x1', 'c'],
                            bounds = {'x1':(-5,5),'c':(-1,1)},
                            correct_mwdust=True, dust_model='F99', ebv=None):
        import sncosmo
        if correct_mwdust:
            if dust_model == 'F99':
                dust = sncosmo.F99Dust()
            elif dust_model == 'CCM89':
                dust = sncosmo.CCM89Dust()
            elif dust_model == 'OD94':
                dust = sncosmo.OD94Dust()
            else:
                raise ValueError(f'{dust_model} is not a valid dust model in SNCosmo.') 
            sncosmo_model = sncosmo.Model(source=source,effects=[dust],
                                          effect_names=['mw'],effect_frames=['obs'])
            
            if dust_model == 'CCM89' or dust_model == 'OD94': 
                sncosmo_model.set(mwr_v=3.1)  # Fix Milky Way R_V at 3.1
            
            # User to provide MW extinction values, computed using RA and Dec combined with sfdmap
            # See guidance at https://sncosmo.readthedocs.io/en/stable/models.html - 'Adding Milky Way dust'
            # if ebv is not None:
            #TODO: discuss
            sncosmo_model.set(mwebv=0)
            
            # else:
            #     raise ValueError('You must provide a Milky Way extinction value if you wish to model dust extinction.')
        
        else:
            sncosmo_model = sncosmo.Model(source=source)
        self.sncosmo_model = sncosmo_model
        self.sncosmo_params = params_to_fit
        self.sncosmo_bounds = bounds
            
    def fit_lc(self,plot=True,preset_vals={},
                  bounds=None,
                  raise_error=False,correct_mwdust=True,dust_model='F99',ebv=None,
                  model_source='salt2',iterate_fit=False,iterative_phase_cut=[-10, 40],**kwargs):
        ''' This requires the SNCosmo package and ATLAS filter info (registered as 'o' and 'c').
        # register filters
        
        Example:
        .. code-block:: python
        
        cyan_wav,cyan_filt = np.loadtxt('filters/ATLAS_cyan.dat',unpack=True)
        orange_wav,orange_filt = np.loadtxt('filters/ATLAS_orange.dat',unpack=True)
        sncosmo.register(sncosmo.Bandpass(orange_wav, orange_filt, name='o'))
        sncosmo.register(sncosmo.Bandpass(cyan_wav, cyan_filt, name='c'))
        
        '''
        import sncosmo
        
        # update data
        self._make_sncosmo_data()
        
        # check if model is prepared
        if (not hasattr(self,'sncosmo_model')):
            self._prep_sncosmo_model(source=model_source,correct_mwdust=correct_mwdust,dust_model=dust_model,ebv=ebv)
            
        if not hasattr(self,'sncosmo_redshift_bounds'):
            self.set_sncosmo_redshift_bounds()
            
        if bounds is None:
            bounds = self.sncosmo_bounds
            
        try:
            params_to_fit = self.sncosmo_model._param_names.copy()
            self.sncosmo_model.set(**preset_vals)
            for key in preset_vals:
                params_to_fit.remove(key)
            
            # if bounds is None:
                # bounds = self.sncosmo_redshift_bounds
            if len(params_to_fit) > 0:
                if iterate_fit == False:
                    modelcov = True
                else:
                    modelcov = False

                guess_amplitude = True
                guess_t0 = True
                bounds.update(self.sncosmo_redshift_bounds)
                result,fitted_model = sncosmo.fit_lc(self.sncosmo_data, 
                                                    self.sncosmo_model, 
                                                    params_to_fit, 
                                                    bounds=bounds,
                                                    guess_amplitude=guess_amplitude,
                                                    guess_t0=guess_t0,
                                                    modelcov=modelcov)
                # If the user wishes to perform an iterative fit, we use the preliminary fit to determine an estimate of the peak time, and cut the phase data around this range
                # We then fit again, using the first set of fitted parameters as priors
                if iterate_fit == True:
                    z = result.parameters[0]
                    prelim_t0 = result.parameters[1]
                    self.sncosmo_data['rf_phase'] = (self.sncosmo_data['mjd']-prelim_t0)/(1+z)  # Convert to rest frame days
                    self.sncosmo_data = self.sncosmo_data[(self.sncosmo_data['rf_phase'] > (iterative_phase_cut[0])) & (self.sncosmo_data['rf_phase'] < (iterative_phase_cut[1]))]
                    result, fitted_model = sncosmo.fit_lc(self.sncosmo_data, 
                                                    fitted_model, 
                                                    params_to_fit, 
                                                    bounds=bounds,
                                                    guess_amplitude=False,
                                                    guess_t0=False,
                                                    modelcov=True)
                self.sncosmo_result = result
            else:
                # just plot
                fitted_model = self.sncosmo_model
                result = None
                self.sncosmo_result = result
                
            if plot:
                # sncosmo.plot_lc(self.sncosmo_data, model=fitted_model, errors=result.errors)
                # plt.show()
                param_dict = dict(zip(fitted_model._param_names,fitted_model.parameters))
                param_text = "\n".join(f"{k}: {v:.3f}" for k, v in param_dict.items())
                mjd_grid = np.linspace(self.mjd.min(),self.mjd.max(),200)
                fig,ax = self.plot_lc()
                for filt in ['orange','cyan']:
                    salt2flux = fitted_model.bandflux(filt[0], mjd_grid, zp=self.zp, zpsys='AB')
                    ax.plot(mjd_grid,salt2flux,c=filt,lw=3,alpha=0.3)
                    if ax.get_ylim()[1] < salt2flux.max():
                        ax.set_ylim(ax.get_ylim()[0],salt2flux.max()*1.1)
                ax.text(0.7,0.95,param_text,transform=plt.gca().transAxes,
                        fontsize=12,ha='left',va='top')

            self.sncosmo_fitted_model = fitted_model
            self._get_sncosmo_residual()
        except Exception as e:
            if raise_error:
                raise e
            print('lc fit failed:',str(e))
            return False
        return True

    def plot_sncosmo_lc(self):
        import sncosmo
        if not hasattr(self,'sncosmo_fitted_model'):
            raise ValueError('Run fit_lc() first')
        sncosmo.plot_lc(self.sncosmo_data, 
                        model=self.sncosmo_fitted_model, 
                        errors=self.sncosmo_result.errors)
        
    def set_sncosmo_redshift_bounds(self,redshift_bounds=(0.01,0.1)):
        self.sncosmo_redshift_bounds = {'z':redshift_bounds}
        
    def _get_sncosmo_residual(self):
        ''' return the residuals from the SNCosmo fit '''
        if not hasattr(self,'sncosmo_fitted_model'):
            raise ValueError('Run fit_salt2() first')
        
        # make model flux
        _flux_model_o = self.sncosmo_fitted_model.bandflux('o', self._mjd[self._filters=='o'], 
                                             zp = self.zp, zpsys='ab')
        _flux_model_c = self.sncosmo_fitted_model.bandflux('c', self._mjd[self._filters=='c'], 
                                             zp = self.zp, zpsys='ab')
        _flux_model = np.ones_like(self._mjd) * np.nan   
        _flux_model[self._filters=='o'] = _flux_model_o
        _flux_model[self._filters=='c'] = _flux_model_c
        
        # take residuals
        self._sncosmo_residual_flux = self._flux - _flux_model
        self.sncosmo_residual_flux = self._sncosmo_residual_flux[self.cuts]
    
    def _calculate_sncosmo_residual_cliplevel(self,sigma_factor=1.0):
        ''' calculate the sigma level for clipping SNCosmo residuals '''
        Nsigma = chauvenet_sigma_level(len(self.sncosmo_residual_flux)+2) * sigma_factor
        cut_level = np.std(self.sncosmo_residual_flux) * Nsigma
        return Nsigma,cut_level
    
    def clip_outliers_sncosmo(self,sigma_minimum=3,sigma_factor=1.0,maxiter=10,verbose=True,plot=True,**kwargs):
        ''' iteratively clip sncosmo residuals.
        
        Inputs:
            sigma_minimum (float): minimum sigma level to clip. Regardless of the number of points, data within this sigma level will be retained.
            sigma_factor (float): factor to multiply the auto-estimated clip level. Don't touch unless you know what you are doing.
            maxiter (int): maximum number of iterations
            verbose (bool): print messages
            plot: plot the light curve after each iteration
            
        Outputs:
            None if the fit failed at any point.
        
        '''
        print(f'Minimum sigma level:{sigma_minimum}: data points within this level will not be clipped.')
        
        # initial fit
        N = len(self.mjd)
        fit_success = self.fit_lc(plot=plot,**kwargs)
        if not fit_success:
            return None

        Nsigma, cut_level = self._calculate_sncosmo_residual_cliplevel(sigma_factor=sigma_factor)
        if Nsigma < sigma_minimum:
            print('Suggested clip level is below the minimum. No clipping will be done.')
            return Nsigma
        
        # iterative clipping
        for _ in range(maxiter):
            self.cut(s=self.cuts,sncosmo_res_max=cut_level,**kwargs)
            N_new = len(self.mjd)
            if N_new == N:
                break
            if verbose:
                print(f'Clipping {N-len(self.mjd)} point(s) at {Nsigma:.2f} sigma (|residual| < {cut_level:.2f})')
            N = N_new
            fit_success = self.fit_lc(plot=plot,**kwargs)
            if not fit_success:
                return None
            Nsigma, cut_level = self._calculate_sncosmo_residual_cliplevel(sigma_factor=sigma_factor)
            if Nsigma < sigma_minimum:
                print('Suggested clip level is below the minimum. No further clipping will be done.')
                return Nsigma
            
    def stats_grouped_by_mjd(self,delta_mjd=0.5,filters=['o','c'],clip=3):
        ''' apply binning and get stats per exposure and stats per bin.
        
        Inputs:
            delta_mjd (float): maximum time difference to combine data points
            filters (list): list of filters to combine
            clip (float): sigma clipping level for the pull calculation.

        Returns:
            df_all (pd.DataFrame): DataFrame containing the stats per exposure.
        '''
        df_all = pd.DataFrame(columns=['MJD','MJD_group','flux','flux_err','F','chi2_phot','clip_mask',
                                       'flux_group_median','flux_group_mean','flux_group_wmean',
                                       'flux_group_wmean_err',
                                       'flux_group_std',
                                       'flux_group_nMAD_median','flux_group_nMAD_mean','flux_group_nMAD_wmean',
                                       'flux_group_n'])
        current_cuts = self.cuts.copy()
        for filt in filters:
            # grab data
            self.cut(s=current_cuts,filters=[filt])
            mjd,flux,fluxerr,chi2dof = self.mjd, self.flux, self.flux_err, self.chi2dof

            # combine flux and fluxerr if mjd is within 1 day
            mjds_grouped = np.array_split(mjd, np.where(np.diff(mjd) > delta_mjd)[0]+1)
            flux_grouped = np.array_split(flux, np.where(np.diff(mjd) > delta_mjd)[0]+1)
            fluxerr_grouped = np.array_split(fluxerr, np.where(np.diff(mjd) > delta_mjd)[0]+1)
            chi2dof_grouped = np.array_split(chi2dof, np.where(np.diff(mjd) > delta_mjd)[0]+1)

            for i in range(len(mjds_grouped)):
                mjd = mjds_grouped[i]
                flux = flux_grouped[i]
                fluxerr = fluxerr_grouped[i]
                chi2dof = chi2dof_grouped[i]
                
                if (fluxerr<=0).sum() > 0:
                    continue
                w = 1/fluxerr**2
                if ~np.isfinite(w.sum()) or w.sum()==0:
                    continue
                
                # do sigma clipping in 
                if clip is not None:
                    pull_all, wmean, mask = sigma_clip_in_pull(flux,fluxerr,sigma_scale=clip)
                else:
                    mask = np.ones_like(flux, dtype=bool)
                
                flux_group_median = np.median(flux[mask])
                flux_group_mean = np.mean(flux[mask])
                flux_group_wmean = wmean#np.average(flux, weights=w)
                flux_group_wmean_err = np.sqrt(1/np.sum(w[mask]))
                flux_group_std = np.std(flux[mask])
                flux_group_nMAD_median = np.median(np.abs(flux - flux_group_median))*1.4826
                flux_group_nMAD_mean = np.median(np.abs(flux - flux_group_mean))*1.4826
                flux_group_nMAD_wmean = np.median(np.abs(flux - flux_group_wmean))*1.4826
                flux_group_n = len(flux)
                mjd_combined = np.average(mjd, weights=w)
                for i in range(len(mjd)):
                    df_all.loc[len(df_all)] = [mjd[i],mjd_combined,flux[i],fluxerr[i],filt,chi2dof[i],mask[i],
                                               flux_group_median,flux_group_mean,flux_group_wmean,
                                               flux_group_wmean_err,
                                               flux_group_std,
                                               flux_group_nMAD_median,flux_group_nMAD_mean,flux_group_nMAD_wmean,
                                               flux_group_n]
            self.cut(s=current_cuts)
        return df_all
    
    def calc_nightly_pull(self,delta_mjd=0.5,filters=['o','c'],pull_clip=3):
        ''' Calculates the pull value for each data point by grouping each night's observation.
        
        The pull value is (flux-flux_wmean)/flux_err, where flux_wmean is the sigma-clipped weighted mean flux per night. 
        To calculate the weighted mean flux, we use the iterative sigma-clipping in the pull space.
        
        Inputs:
            pull_clip (float): sigma clipping level for the pull calculation.
        '''
        df_stats = self.stats_grouped_by_mjd(delta_mjd=delta_mjd,filters=filters,clip=pull_clip)
        df_stats['flux_pull'] = (df_stats['flux'] - df_stats['flux_group_wmean']) / df_stats['flux_err']
        
        # update df_phot
        self.df_phot = self.df_phot.merge(df_stats[['MJD','MJD_group','flux_group_wmean','flux_pull','clip_mask']],on='MJD',how='left')
        self.df_phot.loc[self.df_phot['clip_mask'].isna(),'clip_mask'] = False
        self._flux_pull = self.df_phot['flux_pull'].values
        self._pull_clip = self.df_phot['clip_mask'].values.astype(bool)
        
            
    def group_by_mjd(self,delta_mjd=0.5,filters=['o','c'],sigma_clip_level=False,min_Ndata_for_clip=4):
        ''' apply binning. Data is first split into different filters and then combined if the time difference is within delta_mjd.
        
        Inputs:
            delta_mjd (float): maximum time difference to combine data points
            filters (list): list of filters to combine
            sigma_clip_level (bool): the level of sigma-clipping per group. If False, no sigma clipping will be done.
            min_Ndata_for_clip (int): minimum number of data points to apply sigma clipping.
        
        '''
        df_combined = pd.DataFrame(columns=['mjd','flux','flux_err','F'])
        current_cuts = self.cuts.copy()
        for filt in filters:
            # grab data
            self.cut(s=current_cuts,filters=[filt])
            mjd,flux,fluxerr = self.mjd, self.flux, self.flux_err

            # combine flux and fluxerr if mjd is within 1 day
            mjds_grouped = np.array_split(mjd, np.where(np.diff(mjd) > delta_mjd)[0]+1)
            flux_grouped = np.array_split(flux, np.where(np.diff(mjd) > delta_mjd)[0]+1)
            fluxerr_grouped = np.array_split(fluxerr, np.where(np.diff(mjd) > delta_mjd)[0]+1)

            for i in range(len(mjds_grouped)):
                mjd = mjds_grouped[i]
                flux = flux_grouped[i]
                fluxerr = fluxerr_grouped[i]
                if sigma_clip_level and (len(mjd) >= min_Ndata_for_clip):
                    clip = ~sigma_clip(flux,sigma=sigma_clip_level).mask
                    mjd = mjd[clip]
                    flux = flux[clip]
                    fluxerr = fluxerr[clip]

                if (fluxerr<=0).sum() > 0:
                    continue
                w = 1/fluxerr**2
                if ~np.isfinite(w.sum()) or w.sum()==0:
                    continue
                flux_combined = np.average(flux, weights=w)
                fluxerr_combined = np.sqrt(1/np.sum(w))
                mjd_combined = np.average(mjd, weights=w)
                df_combined.loc[len(df_combined)] = [mjd_combined,flux_combined,fluxerr_combined,filt]
            
            self.cut(s=current_cuts)

        return AtlasBinnedPhotometry(df_combined,objname=self.objname,
                                     warn_template_changes=self.warn_template_changes,
                                     zp=self.zp)
    
    def get_zero_flux_obs(self,pkmjd,filt,phase_before=-40,phase_after=200):
        ''' get the fluxes where they are expected to be zero. Use this to correct ill-subtraction.
        
        Inputs:
            pkmjd (float): peak MJD
            filt (str): filter to use ('o' or 'c')
            phase_before (float): phase before peak where the flux should be zero
            phase_after (float): phase after peak where the flux should be zero
        '''
        cuts = self.cuts.copy()
        self.cut(s=cuts,filters=[filt],mjd_max=pkmjd+phase_before)
        flux_before = self.flux
        flux_err_before = self.flux_err
        
        self.cut(s=cuts,filters=[filt],mjd_min=pkmjd+phase_after)
        flux_after = self.flux
        flux_err_after = self.flux_err
        
        self.cut(s=cuts)
        
        zero_flux = np.concatenate([flux_before,flux_after])
        zero_flux_err = np.concatenate([flux_err_before,flux_err_after])
        return zero_flux,zero_flux_err
    
    def get_zero_levels(self,pkmjd,filters=['o','c'],
                       phase_before=-40,phase_after=200):
        ''' A wrapper to get the weighted-average zero level of the light curve. This is useful for correcting ill-subtraction.
        
        Inputs:
            filters (list): list of filters to use
        '''
        zero_levels = {}
        zero_level_errors = {}
        for filt in filters:
            zero_flux,zero_flux_err = self.get_zero_flux_obs(
                pkmjd=pkmjd,
                filt=filt,
                phase_before=phase_before,
                phase_after=phase_after
                )
            if len(zero_flux) == 0 or np.isfinite(zero_flux).sum() == 0:
                zero_levels[filt] = 0
                zero_level_errors[filt] = 0
                continue
            arr_masked = sigma_clip(zero_flux)
            s = ~arr_masked.mask
            zero_levels[filt] = np.average(zero_flux[s],weights=1/zero_flux_err[s]**2)
            zero_level_errors[filt] = np.sqrt(1/np.sum(1/zero_flux_err[s]**2))
        return zero_levels,zero_level_errors
        
    def add_error_quadrature(self,percent_flux=5):
        ''' add flux error in quadrature '''
        self._flux_err = np.sqrt(self._flux_err**2 + (percent_flux/100*self._flux)**2)
        self.cut(s=self.cuts)
    
class AtlasBinnedPhotometry(AtlasPhotometry):
    def __init__(self,df_combined,objname='',
                 warn_template_changes=True,**kwargs):
        self._mjd = df_combined['mjd'].values
        self._flux = df_combined['flux'].values
        self._flux_err = df_combined['flux_err'].values
        self._filters = df_combined['F'].values
        self.objname = objname
        self.df_phot = df_combined
        self.cut()
        self.check_template_change(warn_template_changes)
        for key,val in kwargs.items():
            setattr(self,key,val)
        
    def correct_zero_levels(self,pkmjd,filters=['c','o'],
                            phase_before=-40,phase_after=200,verbose=False):
        zero_levels,zero_level_errors = self.get_zero_levels(
            pkmjd=pkmjd,
            filters=filters,
            phase_before=phase_before,
            phase_after=phase_after
            )
        for filt in filters:
            if verbose:
                print(f'Zero-level correction for {filt}: {zero_levels[filt]:.2f} +/- {zero_level_errors[filt]:.2f}')
            s = self._filters == filt
            self._flux[s] -= zero_levels[filt]
            self._flux_err[s] = np.sqrt(self._flux_err[s]**2 + zero_level_errors[filt]**2)
            
        cuts = self.cuts.copy()
        self.cut(s=cuts)
        return zero_levels,zero_level_errors
        

from scipy.stats import norm

def init_sncosmo_filters(force=True):
    import sncosmo
    import importlib.resources as ir
    print('Adding ATLAS filters to SNCosmo...',end='')
    filename_cyan = ir.files(__package__).joinpath("ATLAS_cyan.dat")
    filename_orange = ir.files(__package__).joinpath("ATLAS_orange.dat")
    cyan_wav,cyan_filt = np.loadtxt(filename_cyan,unpack=True)
    orange_wav,orange_filt = np.loadtxt(filename_orange,unpack=True)
    sncosmo.register(sncosmo.Bandpass(orange_wav, orange_filt, name='o'),force=force)
    sncosmo.register(sncosmo.Bandpass(cyan_wav, cyan_filt, name='c'),force=force)
    print(' Done')

def chauvenet_sigma_level(N):
    ''' calculate the max sigma-level expected at which a data point is observed for given N data points'''
    # Calculate the probability threshold according to Chauvenet's criterion
    threshold_prob = 1 / (4 * N)
    
    # Calculate the sigma level (number of standard deviations)
    sigma_level = norm.ppf(1 - threshold_prob)
    
    return sigma_level

def sigma_clip_in_pull(flux,flux_err,sigma_scale=3):
    """
    sigma_clip_in_pull: sigma-clip the flux in pull space in a iterative process
    """
    indices = np.arange(len(flux))
    mask = np.isfinite(flux) & np.isfinite(flux_err)
    if mask.sum() < 1:
        return np.nan,np.nan,np.nan
    mean = np.average(flux[mask],weights=1/flux_err[mask]**2)
    pull = (flux-mean)/flux_err

    for _ in range(mask.sum()-2):
        # try removing data with the largest pull
        s = mask.copy()
        s[np.where(pull == pull[mask].max())[0][0]] = False
        mean = np.average(flux[s],weights=1/flux_err[s]**2)
        pull = (flux - mean) / flux_err
        
        # apply sigma-clip with new pull and std
        # see if it changes the clip result
        new_mask = np.abs(pull) < max(1,np.std(pull[s]))*sigma_scale
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    mean = np.average(flux[mask],weights=1/flux_err[mask]**2)
    pull_all = (flux - mean)/flux_err
    
    return pull_all, mean, mask