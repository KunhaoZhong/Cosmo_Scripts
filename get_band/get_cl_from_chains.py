import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys, platform, time
import camb
from camb.dark_energy import DarkEnergyPPF
from camb import model, initialpower
from scipy.interpolate import interp1d
import getdist
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Cl_range = np.arange(3, 151) # range of Cls

def get_cl(cosmo_params, param_names):

    params_and_Cls = np.zeros((len(cosmo_params), len(cosmo_params[0])+len(Cl_range)), dtype=np.float64)


    for i in range(len(cosmo_params)):
        Cls = np.zeros(len(Cl_range), np.float64)
        index_theta_MC_100    = int(np.where(param_names=='theta_MC_100')[0]) - 1 # minus one for the #sign header
        index_omegabh2        = int(np.where(param_names=='omegabh2')[0]) - 1 
        index_omegach2        = int(np.where(param_names=='omegach2')[0]) - 1 
        index_tau             = int(np.where(param_names=='tau')[0]) - 1 
        index_As              = int(np.where(param_names=='As')[0]) - 1 
        index_ns              = int(np.where(param_names=='ns')[0]) - 1 
        index_g0_ppf          = int(np.where(param_names=='g0_ppf')[0]) - 1 

        pars = camb.CAMBparams()
        pars.set_cosmology(thetastar = cosmo_params[i][index_theta_MC_100] / 100, ombh2=cosmo_params[i][index_omegabh2], 
                    omch2=cosmo_params[i][index_omegach2], mnu=0.06, omk=0, 
                    tau=cosmo_params[i][index_tau])
        pars.InitPower.set_params(As=cosmo_params[i][index_As], ns=cosmo_params[i][index_ns], r=0)
        pars.set_for_lmax(2500, lens_potential_accuracy=0);
        pars.DarkEnergy = DarkEnergyPPF(w= -1.0, wa=0.0, g0_ppf = cosmo_params[i][index_g0_ppf], c_g_ppf = 0.01) #don't forget to set other DE pars here
        pars.set_accuracy(AccuracyBoost=1.1)
        pars.Accuracy.lSampleBoost = 50
        
        results = camb.get_results(pars) 
        totCL   = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
        Cl_TT   = totCL[:,0]
        Cls     = Cl_TT[Cl_range[0]-1 : Cl_range[-1]]

        params_and_Cls[i] =  np.append(cosmo_params[i], Cls)

        #for monitoring progress of code running
        if i%100 == 0:
            print('rank', rank, "in progress: ", i)
        if i == len(cosmo_params):
            print('rank', rank, "in progress: ", i)
    
    return params_and_Cls





if rank == 0:
    # read thinned chain file and combine them together as name/value
    path = "/gpfs/home/kuzhong/work/cocoa_AS2/Cocoa/projects/AStress/chains/"
    filenames = [path+'cmb_1.post.scale_thin.1.txt', 
                 path+'cmb_1.post.scale_thin.2.txt',
                 path+'cmb_1.post.scale_thin.3.txt',
                 path+'cmb_1.post.scale_thin.4.txt']
    for j, names in enumerate(filenames):
    # Open each file in read mode
        with open(names) as infile:
            if j == 0:
                param_names_str = infile.readline().split()
                param_names = np.asarray(param_names_str)
                params = np.loadtxt(infile)
            else:
                params = np.concatenate((params, np.loadtxt(infile) ))

    # count: the size of each sub-task
    ave, res = divmod(len(params), size)
    count = [ave + 1 if p < res else ave for p in range(size)]
    count = np.array(count)
    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(size)]
    displ = np.array(displ)
    print("Number of cosmologies:  ",len(params))

else:
    param_names = None
    params = None
    # initialize count on worker processes
    count = np.zeros(size, dtype=np.int64)
    displ = np.zeros(size, dtype=np.int64)

param_names = comm.bcast(param_names, root=0)
params      = comm.bcast(params, root=0)
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)

##Scatterv is not working, kinda tricky for 2d array(https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array). 
##One solution is to flatten the 2d array and use the normal scatter and gather mpi functions
##Alternatively, we can write manually:
my_params = params[displ[rank]: displ[rank]+count[rank]]


my_result_Cl = np.zeros((len(my_params), len(my_params[0])+len(Cl_range)), dtype=np.float64)

result_Cl    = np.zeros((len(params),    len(params[0])+len(Cl_range)), dtype=np.float64)

my_result_Cl = get_cl(my_params, param_names)

#result_Cl[displ[rank]: displ[rank]+count[rank]] = my_result_Cl

comm.Barrier()

#Write the header
# if rank==0:
#     header = '              '.join(param_names_str)
#     with open('./projects/AStress/Cl_band/out_Cl.txt', 'w') as outfile:
#         outfile.write(header)
#         for ll in Cl_range:
#             outfile.write("             Cl_" + str(ll))


if rank == 0:
    header = '              '.join(param_names_str)
    for i in Cl_range:
        header =  header + '          Cl_' + str(i)
    with open('./projects/AStress/Cl_band/out_Cl.txt', 'w') as outfile:
        outfile.write(header)

comm.Barrier()

#Write the output, due to parallezation, the order is not the same as the origional chain file.

with open('./projects/AStress/Cl_band/out_Cl.txt', 'ab') as outfile:
    outfile.write(b"\n")
    np.savetxt(outfile, my_result_Cl, delimiter='          ')




# comm.Gatherv(my_result_Cl, [result_Cl, count, displ, MPI.DOUBLE], root = 0)

# # comm.Reduce(result_Cl.tolist(), None,
# #              op=MPI.SUM, root=0)



# if rank == 0:
#     header = '              '.join(param_names_str)
#     for i in Cl_range:
#         header =  header + '          Cl_' + str(i)
#     np.savetxt('./projects/AStress/Cl_band/out_Cl.txt', result_Cl, delimiter='          ', header = header, comments="")





