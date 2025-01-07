import logging
import os

from yaml import CLoader as Loader
from yaml import load

import slcl1butils


def get_conf():
    """

    Returns:
        conf: dict
    """
    local_config_pontential_path = os.path.join(
        os.path.dirname(slcl1butils.__file__), "localconfig.yaml"
    )

    if os.path.exists(local_config_pontential_path):
        logging.debug("local config used")
        config_path = local_config_pontential_path
    else:
        logging.debug("default config used")
        config_path = os.path.join(os.path.dirname(slcl1butils.__file__), "config.yaml")
    logging.debug("config path: %s", config_path)
    stream = open(config_path, "r")
    conf = load(stream, Loader=Loader)
    return conf
