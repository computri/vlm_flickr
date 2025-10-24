from pathlib import Path
from omegaconf import OmegaConf
import sys, re

def load_cfg():

    cfgs = [
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load("configs/dataset/flickr.yaml"),
        OmegaConf.load("configs/model/default.yaml"),
        OmegaConf.load("configs/train/default.yaml"),
    ]

    # CLI overrides
    dotlist = [re.sub(r"^--", "", a) for a in sys.argv[1:] if "=" in a]
    if dotlist:
        cfgs.append(OmegaConf.from_dotlist(dotlist))


    # Merge CLI overrides
    cfg = OmegaConf.merge(*cfgs)#, OmegaConf.from_cli())

    # cfg = OmegaConf.merge(cfg, OmegaConf.from_cli()) 
    return cfg