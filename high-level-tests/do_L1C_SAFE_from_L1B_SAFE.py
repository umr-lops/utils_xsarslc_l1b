import os, sys
import pdb
from glob import glob
from slcl1butils.scripts.do_IW_L1C_SAFE_from_L1B_SAFE import do_L1C_SAFE_from_L1B_SAFE
import slcl1butils.utils
from importlib import reload
from slcl1butils.utils import get_test_file
import logging
reload(logging)
logging.basicConfig(level=logging.INFO)
from slcl1butils.get_config import get_conf
conf = get_conf()
conf
one_safe_l1b = get_test_file('S1B_IW_XSP__1SDV_20210328T055258_20210328T055325_026211_0320D4_DC31_A13.SAFE')
one_safe_l1b
ancillary_datasets = conf['auxilliary_dataset']
ancillary_datasets.pop('ww3hindcast_spectra',None)
ancillary_datasets.pop('ww3_global_yearly_3h',None)
full_safe_files = [one_safe_l1b]
version = '0.1'
for ffi,full_safe_file in enumerate(full_safe_files):

    print('%i/%i'%(ffi,len(full_safe_files)))
    print('===')
    print(os.path.basename(full_safe_file))
    print('===')
    ret = do_L1C_SAFE_from_L1B_SAFE(full_safe_file,version=version,outputdir=conf['iw_outputdir'],
                                    ancillary_list=ancillary_datasets,dev=True,overwrite=True)
    logging.info('new file: %s',ret)
logging.info('high level check: OK (successful)')