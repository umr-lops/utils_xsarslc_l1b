#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
"""
import logging
import subprocess

import numpy as np

if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--version", type=str, help="version of the run e.g. 1.5", required=True
    )
    parser.add_argument("--outputdir", action="store", default=None, required=False)
    # parser.add_argument('--mode', choices=['range_dates','monthly'], default='range_dates',
    #                     help='mode : range_dates (last 20 days) or monthly (last 12 months)')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    prunexe = "/appli/prun/bin/prun"
    logging.info("outputdir : %s", args.outputdir)
    # listing = '/home1/scratch/agrouaze/listing_wv_l1b_1.4b_nc.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/wv_safe_L1B_L1C.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/wv_safe_L1B2.0_L1C.txt'
    listing = "/home/datawork-cersat-public/project/sarwave/data/listings/wv_safe_L1B2.4_L1C.txt"
    # listing = '/home1/scratch/agrouaze/very_small_listing_wv_for_test_extract_from_L1B2.0.txt' # when you want to test the prun
    # modify initial listing with 2 more args
    listing2 = listing + ".modified"
    fid = open(listing2, "w")
    content = open(listing).readlines()
    taille = len(content)
    for ll in content:
        ll2 = ll.replace("\n", "") + " " + args.version + " " + args.outputdir + "\n"
        fid.write(ll2)
    fid.close()
    # pbs = '/home1/datahome/agrouaze/sources/git/utils_xsarslc_l1b/l1butils/scripts/do_WV_L1C_SAFE_from_L1B_SAFE.pbs'
    pbs = "/home1/datahome/agrouaze/sources/git/utils_xsarslc_l1b/slcl1butils/scripts/do_WV_L1C_SAFE_from_L1B_SAFE.pbs"

    # call prun
    opts = " --split-max-lines=%s --background -e " % (
        np.ceil(taille / 9900.0).astype(int)
    )  # to respect prun constraint on the number max of sublistings 10000
    # if taille < 9999:
    #     opts = ' --split-max-lines=1 --background -e '
    # else:
    #     opts = ' --split-max-lines=3 --background -e '
    # cmd = prunexe+opts+pbs+' '+args.xspeconfigname+' '+args.version+' '+listing
    cmd = prunexe + opts + pbs + " " + listing2
    logging.info("cmd to cast = %s", cmd)
    st = subprocess.check_call(cmd, shell=True)
    logging.info("status cmd = %s", st)
