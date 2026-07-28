import logging
import os

from yaml import CLoader as Loader
from yaml import load

import slcl1butils

logger = logging.getLogger(__name__)


def get_conf(path_config_input=None):
    """

    Args:
        path_config (str): full path of the configuration .yaml file

    Returns:
        conf: dict
    """
    if path_config_input:
        config_path = path_config_input
        logger.debug("path for config: %s", config_path)
    else:
        local_config_pontential_path = os.path.join(
            os.path.dirname(slcl1butils.__file__), "localconfig.yaml"
        )

        if os.path.exists(local_config_pontential_path):
            logger.debug("local config used")
            config_path = local_config_pontential_path
        else:
            logger.debug("default config used")
            config_path = os.path.join(
                os.path.dirname(slcl1butils.__file__), "config.yaml"
            )
    logger.debug("config path: %s", config_path)
    stream = open(config_path, "r")
    conf = load(stream, Loader=Loader)
    return conf


def get_product_id_parameters(file_path_products_description, product_id):
    """

    Args:
        file_path_products_description: str full path
    Returns:
        conf: dict
    """
    stream = open(file_path_products_description, "r")
    conf = load(stream, Loader=Loader)
    return conf[product_id]
