


nonlinear_l2509 = {
'likelihood': {'planck_2018_highl_plik.TT': 
{'clik_file':'/gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules/data/planck/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik'}},

 'theory': {  'classy': {'path': '/gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules/code/class_public',
                      'extra_args': {'N_ncdm': 1,
                                      'N_ur': 2.0328,
                                      'non linear': 'halofit'}},
                                      },
 'params': {
 'A_s': 2.1005829616811546e-09, 
 'n_s': 0.96605, 
 'H0': 67.32, 
 'omega_b': 0.022383, 
 'omega_cdm': 0.12011, 
 'm_ncdm': 0.06, 
 'tau_reio': 0.0543,  
 },
                                      }

from cobaya.yaml import yaml_load

# Add your external packages installation folder
nonlinear_l2509['packages_path'] = '/gpfs/home/kuzhong/work/cocoa1/Cocoa/external_modules'

from cobaya.model import get_model

model = get_model(nonlinear_l2509)