import pandas as pd
import numpy as np
from .api_ATLAS import AtlasPhotometry
from astropy.io import fits
from astropy.table import Table
def _parse_metadata_line(line):
    # remove comments
    line = line.split('#')[0]
    # try:
    # key,val = line.split(':')
    # except Exception:
    key = line.split(':')[0]
    val = ''.join(line.split(':')[1:])
    
    # strip the first whitespace
    val = ' '.join(val.split())
    return key,val

def read_SNANA_data(path):
    
    with open(path) as f:
        lines = f.readlines()
        
        values = []
        metadata = {}
        for line in lines:
            if len(line.split()) <= 1:
                continue
            if line.startswith('#'):
                continue
            elif line.startswith('VARLIST:'):
                columns = line.split()[1:]
            elif 'NOBS:' in line:
                metadata['NOBS'] = int(line.split()[1])
            elif line.startswith('OBS:'):
                values.append(line.split()[1:])
            # elif '+-' in line:
            #     key,val,val_err = line.split('#')[0].replace(':',' ').replace('+-',' ').split()
            #     metadata[key] = val
            #     metadata[key+'_err'] = val_err
            elif ':' in line:
                key,val = _parse_metadata_line(line)
                metadata[key] = val
                
    # parse into pandas dataframe
    df_data = pd.DataFrame(values,columns=columns)
    for key,val in metadata.items():
        df_data.attrs[key] = val
        
    return df_data

class SNANAPhotometry(AtlasPhotometry):
    def __init__(self,path=None,objname='',zp=27.5,df_phot=None):
        if df_phot is not None:
            self.df_phot = df_phot.copy()
        else:
            self.df_phot = read_SNANA_data(path)
        self.metadata = self.df_phot.attrs
        self._mjd = self.df_phot['MJD'].values.astype(float)
        self._flux = self.df_phot['FLUXCAL'].values.astype(float)
        self._flux_err = self.df_phot['FLUXCALERR'].values.astype(float)
        self.objname = objname
        self.zp = zp
        
        if 'FLT' not in self.df_phot.columns:
            _filters = self.df_phot['BAND']
        else:
            _filters = self.df_phot['FLT']
            _filters = _filters.apply(lambda x: x.rsplit('/')[-1])
        self._filters = _filters.values.astype(str)
        self._chi2dof = np.ones_like(self._mjd) * 0
        self.cut()

    def _format_df_phot(self):
        pass
    
    def match_obs(self, phot_obs):
        s = self._mjd == phot_obs._mjd
        self.cut(s=s)
    
    
# class SnanaSimPhotometry(AtlasPhotometry):
#     def __init__(self, df_phot,objname='',zp=27.5)

class SNANASim():
    def __init__(self,head_file,phot_file):
        self.head_file = head_file
        self.phot_file = phot_file
        hdul_head = fits.open(head_file)
        hdul_phot = fits.open(phot_file)
        df_header = Table(hdul_head['Header'].data).to_pandas()
        df_photometry = Table(hdul_phot['Photometry'].data).to_pandas()
        self.df_header = df_header
        self.df_photometry = df_photometry
        
    def get(self,simid,objname=None):
        if objname is None:
            objname = f'SIM {simid}'

        ptrobs_min,ptrobs_max = self.df_header.loc[self.df_header['SNID'].str.strip().astype(int).eq(simid),['PTROBS_MIN','PTROBS_MAX']].values.T
        df_phot = self.df_photometry.loc[np.arange(ptrobs_min-1,ptrobs_max,1)]
        df_phot['FLT'] = df_phot['BAND'].str.strip().apply(lambda x: x[-1])
        df_phot = df_phot[~df_phot['MJD'].duplicated(keep='last')]

        return SNANAPhotometry(objname=objname,df_phot=df_phot.copy())


class SNANAData():
    def __init__(self,head_file,phot_file):
        self.head_file = head_file
        self.phot_file = phot_file
        hdul_head = fits.open(head_file)
        hdul_phot = fits.open(phot_file)
        df_header = Table(hdul_head['Header'].data).to_pandas()
        df_photometry = Table(hdul_phot['Photometry'].data).to_pandas()
        self.df_header = df_header
        self.df_photometry = df_photometry
        self.unique_ids = df_header['SNID'].str.strip().unique()
        
    def get(self,simid,objname=None):
        if objname is None:
            objname = f'SIM {simid}'

        ptrobs_min,ptrobs_max = self.df_header.loc[self.df_header['SNID'].str.strip().eq(simid),['PTROBS_MIN','PTROBS_MAX']].values.T
        df_phot = self.df_photometry.loc[np.arange(ptrobs_min-1,ptrobs_max,1)]
        df_phot['FLT'] = df_phot['BAND'].str.strip().apply(lambda x: x[-1])
        df_phot = df_phot[~df_phot['MJD'].duplicated(keep='last')]

        return SNANAPhotometry(objname=objname,df_phot=df_phot.copy())
