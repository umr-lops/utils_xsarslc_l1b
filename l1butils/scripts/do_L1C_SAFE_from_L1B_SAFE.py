
import argparse
from l1butils.raster_readers import ecmwf_0100_1h
from l1butils.raster_readers import ww3_global_yearly_3h
from l1butils.raster_readers import resource_strftime
from scipy.interpolate import RectBivariateSpline
from datetime import datetime, timedelta
from shapely import wkt
from glob import glob
import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
from shapely import geometry
from datatree import DataTree
import sys, os
from l1butils.get_polygons_from_l1b import  get_swath_tiles_polygons_from_l1bgroup
from l1butils.utils import timing, haversine, map_blocks_coords
from l1butils.coloc.coloc import raster_cropping_in_polygon_bounding_box, coloc_tiles_from_l1bgroup_with_raster
from l1butils.compute.compute_from_l1b import compute_xs_from_l1b
from l1butils.compute.cwave import compute_cwave_parameters
from l1butils.compute.macs import compute_macs



def do_L1C_SAFE_from_L1B_SAFE(full_safe_file):
        
    # Ancillary data to be colocated
    ancillary_ecmwf = {}
    ancillary_ecmwf['resource'] = '/home/datawork-cersat-public/provider/ecmwf/forecast/hourly/0100deg/netcdf_light/%Y/%j/ECMWF_FORECAST_0100_%Y%m%d%H%M_10U_10V.nc'
    ancillary_ecmwf['step'] = 1
    ancillary_ecmwf['name'] = 'ecmwf_0100_1h'

    ancillary_ww3 = {}
    ancillary_ww3['resource'] = '/home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/%Y/FIELD_NC/LOPS_WW3-GLOB-30M_%Y%m.nc'
    ancillary_ww3['step'] = 3
    ancillary_ww3['name'] = 'ww3_global_yearly_3h'

    #ancillary_list = [ancillary_ecmwf]#,ancillary_ww3]
    ancillary_list = [ancillary_ecmwf,ancillary_ww3]
        
        
    #
    safe_file = os.path.basename(full_safe_file)
    safe_path = os.path.dirname(full_safe_file) + '/'
        
    # Processing Parameters:
    cwave = True; macs = True; colocat = True
    sth = macs*cwave
    
    files = glob(safe_path+safe_file+'/'+'*_L1B_*nc')
    if len(files)==0:
        return None
    #print(files)
    
    # Loop on L1B netCDF files (per slice)
    for _file in files:

        print('')
        print('File in: ', _file)
        
        #====================
        # X-SPEC 
        #====================
        #
        # Intraburst at 2tau x-spectra
        burst_type='intra';time_separation='2tau'
        xs_intra,ds_intra = compute_xs_from_l1b(_file,burst_type=burst_type,time_separation=time_separation)
        # Interburst x-spectra
        burst_type='inter';time_separation='None'
        xs_inter,ds_inter = compute_xs_from_l1b(_file,burst_type=burst_type,time_separation=time_separation)
    
        #====================
        # CWAVE 
        #====================    
        if (cwave):
            #
            # CWAVE Processing Parameters
            kmax = 2 * np.pi / 25 ; kmin = 2 * np.pi / 600
            Nk=4; Nphi=5
            #
            # Intraburst at 2tau CWAVE parameters
            ds_cwave_parameters_intra = compute_cwave_parameters(xs_intra, save_kernel=False,  kmax=kmax, kmin=kmin, Nk=Nk, Nphi=Nphi)
            # Interburst CWAVE parameters
            ds_cwave_parameters_inter = compute_cwave_parameters(xs_inter, save_kernel=False,  kmax=kmax, kmin=kmin, Nk=Nk, Nphi=Nphi)
            # updating the L1B dataset
            ds_intra = xr.merge([ds_intra,ds_cwave_parameters_intra])
            ds_inter = xr.merge([ds_inter,ds_cwave_parameters_inter])
            
        #====================
        # MACS 
        #====================
        if (macs):
            # MACS parameters
            lambda_range_max = [50,75,100,125,150,175,200,225,250,275]
            # Intraburst at 2tau MACS
            ds_macs_intra = compute_macs(xs_intra, lambda_range_max = lambda_range_max)
            # Interburst MACS
            ds_macs_inter = compute_macs(xs_inter, lambda_range_max = lambda_range_max)
            # updating the L1B dataset
            ds_intra = xr.merge([ds_intra,ds_macs_intra])
            ds_inter = xr.merge([ds_inter,ds_macs_inter])
            
            
            
        #====================
        # COLOC
        #====================
        if (colocat):            
            
            ds_inter_list=[]; ds_intra_list=[]
            for ancillary in ancillary_list:
            
                #print(ancillary['name'])

                # For each L1B
                #burst_type = 'intra' 
                #l1b_ds = xr.open_dataset(_file,group=burst_type+'burst')
                
                #===========================================
                ## Check if the ancillary data can be found
                sar_date = datetime.strptime(str.split(ds_intra.attrs['start_date'],'.')[0],'%Y-%m-%d %H:%M:%S')
                closest_date, filename =  resource_strftime(ancillary['resource'],step=ancillary['step'],date=sar_date)
                print(filename)
                if (len(glob(filename))!=1):
                    continue
                # Getting the raster from anxillary data
                if (ancillary['name'] == 'ecmwf_0100_1h'):
                    raster_ds = ecmwf_0100_1h(filename)
                if (ancillary['name'] == 'ww3_global_yearly_3h'):
                    raster_ds = ww3_global_yearly_3h(filename, closest_date)

                # Get the polygons of the swath data
                polygons = get_swath_tiles_polygons_from_l1bgroup(ds_intra,swath_only=True)
                # Crop the raster to the swath bounding box limit
                raster_bb_ds = raster_cropping_in_polygon_bounding_box(polygons['swath'][0], raster_ds)

                # Loop on the grid in the product
                burst_types = ['intra', 'inter'] 
                for burst_type in burst_types:
                    # Define the dataset to work on
                    # get the mapped raster onto swath grid for each tile
                    if (burst_type == 'intra'): 
                        #l1b_ds_intra = xr.open_dataset(_file,group=burst_type+'burst')
                        #_ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_intra, raster_bb_ds, apply_merging=False)
                        #ds_intra_list.append(_ds)
                        _ds_intra = coloc_tiles_from_l1bgroup_with_raster(ds_intra, raster_bb_ds, apply_merging=False)
                        #ds_intra_list.append(_ds_intra)
                        # Merging the datasets
                        ds_intra = xr.merge([ds_intra, _ds_intra])
                    else:
                        #l1b_ds_inter = xr.open_dataset(_file,group=burst_type+'burst')
                        #_ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_inter, raster_bb_ds, apply_merging=False)
                        #ds_inter_list.append(_ds)
                        _ds_inter = coloc_tiles_from_l1bgroup_with_raster(ds_inter, raster_bb_ds, apply_merging=False)
                        #ds_inter_list.append(_ds_inter)
                        # Merging the datasets
                        ds_inter = xr.merge([ds_inter, _ds_inter])
                        
            # Merging the datasets
            #ds_intra = xr.merge([ds_intra, xr.merge(ds_intra_list)])
            #ds_inter = xr.merge([ds_inter, xr.merge(ds_inter_list)])
                        
        #====================
        # FILE OUT
        #====================        
        #sth=0
        if (sth>0):
            #
            #
            # Output file directory
            pathout_root = safe_path.replace('l1b','l1c')
            #print(~os.path.isdir(pathout_root))
            if (os.path.isdir(pathout_root)==False):
                os.system('mkdir ' + pathout_root)
            pathout = pathout_root + safe_file +'/'
            #print(os.path.isdir(pathout))
            if (os.path.isdir(pathout)==False):
                os.system('mkdir ' + pathout)
            #
            # Ouput filename
            fileout = os.path.basename(_file).replace('L1B','L1C')
            print('File out: ', pathout+fileout)
            #
            # Arranging & saving Results
            # Building the output datatree
            dt = DataTree()
            burst_type='intra';
            dt[burst_type+'burst'] = DataTree(data=ds_intra)
            burst_type='inter'
            dt[burst_type+'burst'] = DataTree(data=ds_inter)
            #
            # SAving the results in netCDF
            dt.to_netcdf(pathout+fileout)

            
        
    return 0






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Script to print the filename')
    # parser.add_argument('--resolution', type = int, help = 'image resolution in meters')
    parser.add_argument('--inputfile', help = 'File complete path')

    args = parser.parse_args()
    # resolution = args.resolution
    full_safe_file = args.inputfile
    
    do_L1C_SAFE_from_L1B_SAFE(full_safe_file)