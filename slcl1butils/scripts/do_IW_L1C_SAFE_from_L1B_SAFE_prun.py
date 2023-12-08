#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
"""
import sys
import os
import subprocess
import logging

if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--outputdir", action="store", default=None, required=False)
    parser.add_argument(
        "--finalstorage",
        required=False,
        help="directory where the L1C will be moved after generation (do not put the version) [optional]",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the existing outputs [default=False]",
        required=False,
    )
    parser.add_argument(
        "--version", type=str, help="version of the run e.g. 1.5", required=True
    )
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
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_1.4k.txt' # 5781 SAFE, 632Go en L1B
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_1.4k_v2.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_1.4k_v3.txt' # the most complete for SAFE 1.4k L1B
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.1.txt' #2km decimated with negative overlap
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.8.txt' # 17.7km all TC 899 SAFE
    # listing = '/home1/scratch/agrouaze/listing_IW_SLC_tiff_ifremer_medium_size_v25aug2023_noduplicate.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/L1B_L1C_listing_safe_missing_IW_1.4k_operational.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/L1B_L1C_listing_safe_missing_IW_1.4j.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_1.4k_test.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.8_v2run_not_finished.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.8_v3.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.9_v1.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.8_v4.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/iw_l1b_safe_2.8_without_l1c_29sept23.txt' # vcoems from check_sar_wave_processing_completness.ipynb
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/case-studies/TC_SURIGAE_2021_IW_XSP_SAFE_L1B_v1.txt'
    # listing = '/home/datawork-cersat-public/project/sarwave/data/listings/case-studies/IW_SLC_Ciaran2023_SAFE_L1B.txt'
    # listing = "/home/datawork-cersat-public/project/sarwave/data/listings/iw_slc_medium_listing_L1B_run3.7_outputs.txt"  # 8960 SAFE
    listing = "/home/datawork-cersat-public/project/sarwave/data/listings/iw_slc_medium_listing_L1B_run3.7_outputs_v2.txt"  # 9012 SAFE
    logging.info("outputdir : %s", args.outputdir)
    # modify initial listing with more args
    listing2 = listing + ".modified"
    fid = open(listing2, "w")
    content = open(listing).readlines()

    cpt_todo = 0
    cpt_already = 0
    if args.finalstorage:
        oo = args.finalstorage
    else:
        oo = args.outputdir
    logging.info("considered outputdir for filtering out existing files is :%s", oo)
    for ll in content:
        treat_it = False

        inpu = ll.replace("\n", "")
        safe_basename = os.path.basename(inpu)

        output_filename1 = os.path.join(
            oo,
            args.version,
            safe_basename,
        )
        output_filename2 = os.path.join(oo, safe_basename)
        if args.overwrite:
            treat_it = True
            cpt_todo += 1
        else:
            if os.path.exists(output_filename1) or os.path.exists(output_filename2):
                cpt_already += 1
            else:
                treat_it = True
                cpt_todo += 1

        if treat_it:
            if args.overwrite:
                ll2 = (
                    ll.replace("\n", "")
                    + " "
                    + args.version
                    + " "
                    + args.outputdir
                    + " --overwrite\n"
                )
            else:
                ll2 = (
                    ll.replace("\n", "")
                    + " "
                    + args.version
                    + " "
                    + args.outputdir
                    + "\n"
                )
            fid.write(ll2)
    fid.close()
    logging.info("counter : already done: %s, todo: %s", cpt_already, cpt_todo)
    didi = os.path.dirname(os.path.realpath(__file__))
    pbs = os.path.join(didi, "do_IW_L1C_SAFE_from_L1B_SAFE.pbs")
    logging.info("pbs file: %s", pbs)
    taille = cpt_todo
    logging.info("Number of SAFE to treat from Level-1B to Level-1C: %s", taille)
    # call prun
    if taille < 9999:
        opts = " --split-max-lines=1 --background -e "
    else:
        opts = " --split-max-lines=3 --background -e "
    cmd = prunexe + opts + pbs + " " + listing2
    logging.info("cmd to cast = %s", cmd)
    st = subprocess.check_call(cmd, shell=True)
    logging.info("status cmd = %s", st)
