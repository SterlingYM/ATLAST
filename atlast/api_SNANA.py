import pandas as pd
from .api_ATLAS import AtlasPhotometry

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
            self.df_phot = df_phot
        else:
            self.df_phot = read_SNANA_data(path)
        self.metadata = self.df_phot.attrs
        self._mjd = self.df_phot['MJD'].values.astype(float)
        self._flux = self.df_phot['FLUXCAL'].values.astype(float)
        self._flux_err = self.df_phot['FLUXCALERR'].values.astype(float)
        self.objname = objname
        self.zp = zp
        
        _filters = self.df_phot['FLT']
        _filters = _filters.apply(lambda x: x.rsplit('/')[-1])
        self._filters = _filters.values.astype(str)
        self.cut()

    def _format_df_phot(self):
        pass
    