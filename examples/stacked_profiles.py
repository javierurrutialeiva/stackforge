from src import *
from configparser import ConfigParser
import argparse
import os 

if __name__ == "__main__":
    basePath = "/scratch/users/javierul/sims.TNG/TNG300-3/output/"
    h = stackforge.halo(sim = "TNG", basePath = basePath, snapNum = 99, haloID = 0) 
    R = np.linspace(0, 1e4, 100)
    h.generate_profiles(R, projection = "2d", use_area = True, triax = True)
    h.generate_profiles(R, projection = "3d", use_area = True, triax = True)
    ne2d = h.profiles2D[0] #index 0 is for electron number density
    ne3d = h.profiles3D[0]
    fig, ax = plt.subplots()
    ax.loglog(h.R_centers, ne2d, label = "2D")
    ax.loglog(h.R_centers, ne3d, label = "3D")
    ax.legend(frameon = False)
    ax.set(xlabel = "R (ckpc)", ylabel = "electron number density")
    fig.savefig("electron_number_density.pdf")