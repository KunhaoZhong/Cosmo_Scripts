from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#add location of data vector file for plotting
d1 = np.genfromtxt("des_y3_theory_3x2pt.modelvector")[:,1]

#use this covariance for redmagic lens sample
covfile = "cov_unblinded_11_13_20.txt"

ndata = d1.shape[0]
m = np.genfromtxt("3x2pt_baseline.mask")[:,1]

data = np.genfromtxt(covfile)
cov = np.zeros((ndata,ndata))
for i in range(0,data.shape[0]):
	cov[int(data[i,0]), int(data[i,1])] = data[i,2]
	cov[int(data[i,1]), int(data[i,0])] = data[i,2]
	cov[int(data[i,0]),int(data[i,1])]*= m[int(data[i,0])]*m[int(data[i,1])]
	cov[int(data[i,1]),int(data[i,0])]*= m[int(data[i,0])]*m[int(data[i,1])] 

#s now contains sqrt(cov[i,i])
s = np.sqrt(np.diag(cov))
ind = np.arange(0, ndata)

nxip = 200
nxim = 200
nw = 100
nggl = ndata - nw - nxip -nxim
s = np.sqrt(np.diag(cov))

fig, axs = plt.subplots(2, 1)
axs[0].set_yscale('log')
axs[0].set_ylim(2.e-6,2.5e-3)
axs[0].set_xlim(nxip+nxim,nxip+nxim+nggl-1)
axs[0].set_ylabel(r'$\gamma_t$', fontsize = 18)
axs[0].errorbar(ind, d1, s, marker='o', color='k',linestyle = '', markersize = 1.00, alpha = 0.25)
axs[0].plot(ind, d1, marker='o', color='r',linestyle = '',markersize = 1.5)
axs[0].set_xticks(np.arange(min(ind[nxip+nxim:nxip+nxim+nggl-1]), max(ind[nxip+nxim:nxip+nxim+nggl-1]), 20.0))
for i in np.arange(min(ind[nxip+nxim:nxip+nxim+nggl-1]), max(ind[nxip+nxim:nxip+nxim+nggl-1]), 20.0):
	axs[0].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)
axs[0].tick_params(axis='x', labelsize=8)

axs[1].set_yscale('log')
axs[1].set_ylim(1.e-4,0.6)
axs[1].set_xlim(nxip+nxim+nggl,ndata)
axs[1].set_ylabel(r'$w$', fontsize = 18)
axs[1].errorbar(ind, d1, s, marker='o', color='k', linestyle = '', markersize = 1.00, alpha = 0.25)
axs[1].plot(ind,d1,marker='o', color='r',linestyle = '',markersize = 1.5)
axs[1].set_xticks(np.arange(min(ind[nxip+nxim+nggl:ndata]), max(ind[nxip+nxim+nggl:ndata]), 20.0))
for i in np.arange(min(ind[nxip+nxim+nggl:ndata]), max(ind[nxip+nxim+nggl:ndata]), 20.0):
	axs[1].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

plt.show()
plt.tight_layout()
plt.savefig("plotdv2.pdf")
