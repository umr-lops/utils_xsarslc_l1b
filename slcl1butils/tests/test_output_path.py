import pytest

from slcl1butils.utils import get_l1c_filepath

inputs_l1b = [
    "/tmp/2022/127/S1A_IW_XSP__1SDV_20220507T162437_20220507T162504_043107_0525DE_B14E_A02.SAFE/l1b-s1a-iw1-xsp-vv-20220507t162439-20220507t162504-043107-0525de-004_a02.nc",
    "/tmp/2022/127/S1A_IW_XSP__1SDV_20220507T162437_20220507T162504_043107_0525DE_B14E_A02.SAFE/l1b-s1a-iw1-xsp-vv-20220507t162439-20220507t162504-043107-0525de-004-a02.nc",
    # "/tmp/data/sentinel1/S1A_IW_SLC__1SDV_20220507T162437_20220507T162504_043107_0525DE_B14E.SAFE/measurement/s1a-iw1-slc-vv-20220507t162439-20220507t162504-043107-0525de-004.tiff",
    "/tmp/S1A_IW_XSP__1SDV_20231104T182308_20231104T182335_051071_06288B_C214.SAFE/s1a-iw1-slc-vv-20231104t182308-20231104t182333-051071-06288b-004_L1B_xspec_IFR_3.3.nc",
]
expected_l1c = [
    "/tmp/2022/127/S1A_IW_XSP__1SDV_20220507T162437_20220507T162504_043107_0525DE_B14E_B02.SAFE/l1c-s1a-iw1-xsp-vv-20220507t162439-20220507t162504-043107-0525de-004-b02.nc",
    "/tmp/2023/308/S1A_IW_XSP__1SDV_20231104T182308_20231104T182335_051071_06288B_C214_B02.SAFE/l1c-s1a-iw1-xsp-vv-20231104t182308-20231104t182333-051071-06288b-004-b02.nc",
]


@pytest.mark.parametrize(
    ["inputs_l1b", "expected_l1c"],
    (
        pytest.param(inputs_l1b[0], expected_l1c[0]),
        pytest.param(inputs_l1b[1], expected_l1c[0]),
        pytest.param(inputs_l1b[2], expected_l1c[1]),
        # pytest.param(inputs_l1slc[1], expected_l1b[0]),
    ),
)
def test_outputfile_path(inputs_l1b, expected_l1c):
    version = "B02"
    outputdir = "/tmp/"
    l1c_full_path = get_l1c_filepath(inputs_l1b, version=version, outputdir=outputdir)

    print(l1c_full_path)
    assert l1c_full_path == expected_l1c
