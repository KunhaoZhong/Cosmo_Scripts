import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys, platform, time
from scipy.interpolate import interp1d
from scipy.integrate import odeint

z_interp_2D = np.linspace(0,2.0,95)
z_interp_2D = np.concatenate((z_interp_2D, np.linspace(2.0,10.0,5)),axis=0)
z_interp_2D[0] = 0

zgs  = np.flip(np.concatenate((z_interp_2D, np.linspace(10.1,1000,3000)),axis=0)) 

def init_G(omegam, omegam_growth, w, w_growth):
      global OM_GROWTH
      global OM_GEO
      global W_GEO
      global W_GROWTH

      OM_GROWTH = omegam_growth
      OM_GEO = omegam
      W_GEO = w
      W_GROWTH = w_growth

def G_GROWTH_ODE(y, N):
      OMG   = OM_GROWTH
      WG    = W_GROWTH
      z     = 1.0/np.exp(N) - 1.0 
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3*(1.0 + WG)) # H^2(z)
      OMG_A = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) / H2                       # Omega_m(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG)) / H2                         # Omega_DE(z)
      HPH   =  -1.5*OMG_A -1.5*OLG_A*(1.0 + WG)                            # dlnH/dlna
      return [y[1], - (4 + HPH)*y[1] - (3.0 + HPH - 1.5 * OMG_A)*y[0]]     # [dG/dlna, d^2G/dlna^2]

def G_GEO_ODE(y, N):
      OMG   = OM_GEO
      WG    = W_GEO
      z     = 1.0/np.exp(N) - 1.0 
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3*(1.0 + WG)) # H^2(z)
      OMG_A = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) / H2                       # Omega_m(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG)) / H2                         # Omega_DE(z)
      HPH   =  -1.5*OMG_A -1.5*OLG_A*(1.0 + WG)                            # dlnH/dlna
      return [y[1], - (4 + HPH)*y[1] - (3.0 + HPH - 1.5 * OMG_A)*y[0]]     # [dG/dlna, d^2G/dlna^2]

def G_ODE_IC_GROWTH(z):
      OMG   = OM_GROWTH
      WG    = W_GROWTH
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3.0*(1.0 + WG)) # H^2(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG))/H2                                  # Omega_DE(z)
      return [1.0, -0.6*(1.0 - WG)*OLG_A]

def G_ODE_IC_GEO(z):
      OMG   = OM_GEO
      WG    = W_GEO
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3.0*(1.0 + WG)) # H^2(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG))/H2                                  # Omega_DE(z)
      return [1.0, -0.6*(1.0 - WG)*OLG_A]




def get_sigma8_split(sigma8_camb):
      sol      = odeint(G_GROWTH_ODE, G_ODE_IC_GROWTH(zgs[0]), np.log(1.0/(1.0 + zgs)))[:,0]
      G_growth = np.flip(sol)[0:len(z_interp_2D)]
      G_growth = G_growth/G_growth[len(G_growth)-1]

      sol2  = odeint(G_GEO_ODE, G_ODE_IC_GEO(zgs[0]), np.log(1.0/(1.0 + zgs)))[:,0]
      G_geo = np.flip(sol2)[0:len(z_interp_2D)]
      G_geo = G_geo/G_geo[len(G_geo)-1]

      split_factor = G_growth[0]**2 / G_geo[0]**2
      return split_factor*sigma8_camb
