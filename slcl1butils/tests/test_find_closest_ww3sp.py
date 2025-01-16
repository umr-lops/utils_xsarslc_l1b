from slcl1butils.coloc.coloc_IW_WW3spectra import find_closest_ww3
import pytest
import os
import slcl1butils
import datetime
from slcl1butils.get_config import get_conf
conf = get_conf()
ww3spectra_hindcast_file = os.path.abspath(os.path.join(os.path.dirname(slcl1butils.__file__),conf['data_dir'],'LOPS_WW3-GLOB-30M_202302_trck.nc'))
cases = {
    'matching':{
        #matching both in time and space:
        'lon':177.5,
        'lat':-34,
        'datesar': datetime.datetime(2023,2,13,17,30),
        'expected': 56
    },
    'no_match_in_space':{
        #not matching in space
        'lon': -50,
        'lat': 40,
        'datesar': datetime.datetime(2023,2,13,17,30),
        'expected':-999
    },
    'no_match_in_time':{
        #not matching in time
        'lon':177.5,
        'lat':-34,
        'datesar': datetime.datetime(2023,2,13,14,29),
        'expected': -999,
    }
}
params_list = []
for case in cases:
    params_list.append((cases[case]['lon'],cases[case]['lat'],cases[case]['datesar'],cases[case]['expected']))

@pytest.mark.parametrize(
    ["lon", "lat","datesar","expected"],
    tuple(params_list),
)
def test_finder(lon,lat,datesar,expected):
    da_index = find_closest_ww3(ww3_path=ww3spectra_hindcast_file,lon=lon,lat=lat,time=datesar)
    actual_index = da_index['WW3spectra_index'].data
    # print('da_index',da_index)
    assert actual_index==expected

if __name__ == '__main__':
    for case in cases:
        test_finder(lon=cases[case]['lon'],lat=cases[case]['lat'],
        datesar=cases[case]['datesar'],expected=cases[case]['expected'])
    print('OK successful test')
