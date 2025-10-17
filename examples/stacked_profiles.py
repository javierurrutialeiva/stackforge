from src import *
from configparser import ConfigParser
import argparse
import os 

parser = argparse.ArgumentParser()

parser.add_argument("--config-file", "-c", default = None, help = "Configuration file.")
args = parser.parse_args()

config_file = args.config_file
current_path = os.path.dirname(os.path.realpath(__file__))

config_filepath = current_path + "/" +config_file
config = ConfigParser()
config.optionxform = str

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

simulation = config["simulation"]["simulation"] #TNG or illustris for the moment

    illustris_config = config["illustris_config"]
    stack_config = config["stack_config"]
    halos_config = config["halos_config"]
    snap_config = config["snap_config"]

    basePath = illustris_config["basePath"]
    Nhalos = int(halos_config["Nhalos"])
    rmin, rmax = prop2arr(stack_config["radius"])
    scale = stack_config["scale"]
    nr = int(stack_config["nradius_bins"])
    R = np.linspace(rmin, rmax, nr) if scale == "linear" else np.logspace(rmin, rmax, nr)
    haloIDs = [i for i in range(Nhalos)]

    projection = stack_config["projection"]
    redshifts = prop2arr(stack_config["redshift"])

    stacks = stack(basePath, None, haloIDs, True, redshift=redshifts[0])
    stacks.compute_1h2hprofiles(R)
    stacks.save()
