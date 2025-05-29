# ATLAST (ATLAS Transient Lightcurve Analysis and Smoothing Tools)
An all-in-one toolkit to download, visualize, and analyze ATLAS light curves

## Usage (note: docs work in progress):
(may be useful for generating SNANA-format data files from ATLAS photometry)

### Installation
~~~bash
$ git clone https://github.com/SterlingYM/ATLAST
$ cd ATLAST
$ pip install -e .
~~~
This installs a toolkit ```atlast``` to your current python environment.

### Query photometry files
~~~python
#### example for querying data ####
from atlas_tools import QueryATLAS
api = QueryATLAS(api_token="your_api_token_here")
task_url = api.request_phot(radeg,decdeg,mjd_min,mjd_max,use_reduced=False) # request photometry
api.save_phot(task_url,folder='phot_output') # save the photometry data in the phot_output folder
~~~

### Inspect light curves
~~~python
#### example for analyzing data & generating SNANA-format files ####
# load photometry & apply binning
phot = AtlasPhotometry('phot_data.txt',objname=CID)
phot.cut(fluxerr_max=2000,chi2dof_max=5) 
phot_binned = phot.group_by_mjd(delta_mjd=0.5,sigma_clip_level=False)
phot_binned.plot_lc()

# remove observations before/after template change if they are far from the peak
# this requires the peak MJD info. You can fit the light curve to get one
phot_binned.cut_data_at_template_change(pkmjd=pkmjd,retain_phase=[-20,50])

# correct zero levels
zero_levels,zero_level_errors = phot_binned.correct_zero_levels(
    pkmjd=pkmjd,
    filters=['c','o'],
    phase_before=-40,
    phase_after=200
    )

# add 5% error in quadrature
phot_binned.add_error_quadrature(percent_flux=5.0)

# phot.to_SNANA can also take a header argument to add metadata to the output file.
headers = {
    'SURVEY': 'ATLAS',
    'SNID': objname,
    'RA': f'{ra:.5f}',
    'DECL': f'{dec:.5f}',
    'RA_HOST': f'{ra_host:.5f}',
    'DEC_HOST': f'{dec_host:.5f}',
    'HOSTGAL_LOGMASS': f'{logmass:.2f} +- {logmass_err:.2f}',
    'REDSHIFT_HELIO': f'{redshift:.6f} +- 0.0001',
    'FILTERS': 'co'
}
phot.to_SNANA('output_file.txt',header=headers)
~~~


NOTE: light curve fitting requires some setup.
1. (required) add ATLAS filters to sncosmo:
~~~python
from atlas_tools import init_sncosmo_filters
init_sncosmo_filters()
~~~

2. (optional) Select the model to fit. The default is 'salt2'. You can also specify here if you want to correct for Milky Way extinction with the ```correct_mwdust``` flag, select from the available dust models (```F99, CCM89, OD94```) using the ```dust``` flag, and provide a predetermined E(B-V) using the ```ebv``` flag. For guidance on how to compute E(B-V) for SNCosmo, see https://sncosmo.readthedocs.io/en/stable/models.html - 'Adding Milky Way dust'.
~~~python
phot._prep_sncosmo_model(source='hsiao',params_to_fit=['z', 't0', 'amplitude'])

# if sources other than 'salt2' are used, the bounds should be adjusted.
phot.plot_lc(bounds=dict_of_tuples)
~~~

3. (optional) set preset (fixed) values. For example, if the redshift, milky-way dust E(B-V), and peak MJD are known:
~~~python
phot.fit_lc(
    preset_vals={
        'mwebv':mwebv,
        'z':redshift,
        't0':pkmjd,
        },
    model_source='salt2',
    correct_mwdust=True,
    raise_error=True,
    plot=True
    )

~~~
It is also possible to perform an initial fit to determine first-pass values for the fitted parameters and then a second iteration of the fit using these values as priors, using the ```iterate_fit``` flag.
