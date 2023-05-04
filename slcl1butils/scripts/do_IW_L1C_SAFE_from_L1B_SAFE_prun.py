#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
"""
import sys
import os
import subprocess
import logging

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='start prun')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--version', type=str,
                        help='version of the run e.g. 1.5', required=True)
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')
    prunexe = '/appli/prun/bin/prun'
    listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_1.4k.txt' # 5781 SAFE, 632Go en L1B
    # modify initial listing with more args
    listing2 = listing + '.modified'
    fid = open(listing2, 'w')
    content = open(listing).readlines()
    taille = len(content)
    for ll in content:
        ll2 = ll.replace('\n', '') + ' ' + args.version + '\n'
        fid.write(ll2)
    fid.close()
    pbs = os.path.join(__file__,'do_IW_L1C_SAFE_from_L1B_SAFE.pbs')
    logging.info('pbs file: %s',pbs)
    # call prun
    if taille < 9999:
        opts = ' --split-max-lines=1 --background -e '
    else:
        opts = ' --split-max-lines=3 --background -e '
    cmd = prunexe + opts + pbs + ' ' + listing2
    logging.info('cmd to cast = %s', cmd)
    st = subprocess.check_call(cmd, shell=True)
    logging.info('status cmd = %s', st)
