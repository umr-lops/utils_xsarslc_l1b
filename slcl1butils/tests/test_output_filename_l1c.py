import pytest

from slcl1butils.utils import get_l1c_filepath

inputs_l1b = [
    "/tmp/data/products/tests/iw/slc/l1b/4.0.0/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058.SAFE/s1b-iw1-slc-vv-20210420t094118-20210420t094144-026549-032b99-004_L1B_xspec_IFR_4.0.0.nc",
    "/tmp/data/products/tests/iw/slc/l1b/4.0.0/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_A03.SAFE/s1b-iw1-slc-vv-20210420t094118-20210420t094144-026549-032b99-004_L1B_xspec_IFR_4.0.0.nc",
]
expected_l1c = [
    "/tmp/2021/110/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_B02.SAFE/l1c-s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004-b02.nc"
]


@pytest.mark.parametrize(
    ["l1b_fullpath", "expected_l1c"],
    (
        pytest.param(inputs_l1b[0], expected_l1c[0]),
        pytest.param(inputs_l1b[1], expected_l1c[0]),
    ),
)
def test_outputfile_path(l1b_fullpath, expected_l1c):
    version = "B02"
    outputdir = "/tmp/"
    l1c_full_path = get_l1c_filepath(l1b_fullpath, version=version, outputdir=outputdir)

    print(l1c_full_path)
    assert l1c_full_path == expected_l1c
