from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#add location of data vector file for plotting
d1 = np.genfromtxt("des_y3_theory_3x2pt.modelvector")[:,1]
d2 = np.genfromtxt("des_y3_theory_3x2pt.modelvector")[:,1]
fracdiff = (d2-d1)/(d1 + 1e-14)

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
axs[0].set_yscale('linear')
axs[0].set_ylim(-1.0,1.0)
axs[0].set_xlim(0, nxip - 1)
axs[0].set_ylabel(r'$\Delta \xi_+/\xi_+$', fontsize = 18)
axs[0].errorbar(ind, fracdiff, s, marker='o', color='k',linestyle = '', markersize = 1.00, alpha = 0.25)
axs[0].plot(ind, fracdiff, marker='o', color='r',linestyle = '',markersize = 1.5)
axs[0].set_xticks(np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0))
for i in np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0):
	axs[0].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

axs[1].set_yscale('linear')
axs[1].set_ylim(-1.0,1.0)
axs[1].set_xlim(nxip, nxip+nxim-1)
axs[1].set_ylabel(r'$\Delta \xi_-/\xi_-$', fontsize = 18)
axs[1].errorbar(ind, fracdiff, s, marker='o', color='k', linestyle = '', markersize = 1.00, alpha = 0.25)
axs[1].plot(ind, fracdiff, marker='o', color='r',linestyle = '',markersize = 1.5)
axs[1].set_xticks(np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0))
for i in np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0):
	axs[1].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

plt.show()
plt.tight_layout()
plt.savefig("plotdv3.pdf")
