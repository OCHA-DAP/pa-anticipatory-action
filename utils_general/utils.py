import zipfile
import logging
import os
import argparse
import requests
import yaml
import coloredlogs

logger = logging.getLogger(__name__)

def parse_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def config_logger(level="INFO"):
    # Colours selected from here:
    # http://humanfriendly.readthedocs.io/en/latest/_images/ansi-demo.png
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        field_styles={
            "name": {"color": 8},
            "asctime": {"color": 248},
            "levelname": {"color": 8, "bold": True},
        },
    )