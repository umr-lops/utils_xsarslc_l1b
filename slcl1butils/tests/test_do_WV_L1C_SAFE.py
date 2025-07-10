import os

# tests/scripts/test_do_WV_L1C_SAFE_from_L1B_SAFE.py
import unittest
from datetime import datetime
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import xarray as xr

# Import the script we want to test
from slcl1butils.scripts import do_WV_L1C_SAFE_from_L1B_SAFE as l1c_script


class TestWV_L1C_Processing(unittest.TestCase):
    """
    Unit tests for the L1B to L1C SAFE processing script.
    Mocks all external dependencies to test the script's logic in isolation.
    """

    def setUp(self):
        """Set up common test data and mock objects."""
        # Create a fake xarray.Dataset to be passed around
        self.fake_ds = xr.Dataset(
            {"dummy_var": (("time",), [1])},
            coords={"time": [np.datetime64("2023-01-01T12:00:00")]},
            attrs={"history": "L1B test data"},
        )
        self.fake_ds_with_corners = self.fake_ds.copy()
        self.fake_ds_with_corners["corner_longitude"] = (
            ("time", "corner"),
            [[-5, -4, -5, -4]],
        )
        self.fake_ds_with_corners["corner_latitude"] = (
            ("time", "corner"),
            [[45, 45, 46, 46]],
        )

        # Fake product configuration
        self.fake_product_config = {
            "ancillary_raster_dataset": ["ecmwf_0100_1h"],
            "crop_xspectra": None,
            "mode": "WV",
            "add_ww3spectra": False,
        }

    def test_get_l1c_filepath_generation(self):
        """
        Verify that the output L1C file path is generated correctly.
        """
        l1b_path = "/data/l1b/S1A_WV1_XSP__1SSV_20230501T055632_20230501T061158_048336_05D036_A24.nc"
        product_id = "B49"
        output_dir = "/tmp/l1c_output"

        with patch(
            "slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.os.makedirs"
        ) as mock_makedirs:
            l1c_path, l1b_version = l1c_script.get_l1c_filepath(
                l1b_path, product_id, outputdir=output_dir
            )

            self.assertEqual(l1b_version, "A24")
            self.assertIn(output_dir, l1c_path)
            self.assertIn(
                os.path.join("2023", "121"), l1c_path
            )  # Check for YYYY/JJJ structure
            self.assertIn(
                "S1A_WV1_XSP__1SSV_20230501T055632_20230501T061158_048336_05D036_B49.nc",
                l1c_path,
            )
            mock_makedirs.assert_called_once()

    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.netcdf_compliant")
    def test_save_l1c_to_netcdf_attributes_and_call(self, mock_netcdf_compliant):
        """
        Ensure the dataset is correctly prepared and saved to NetCDF.
        """
        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.attrs = {}
        mock_netcdf_compliant.return_value = mock_ds

        l1c_path = "/tmp/output.nc"
        product_id_l1c = "B49"
        product_id_l1b = "A24"

        l1c_script.save_l1c_to_netcdf(l1c_path, mock_ds, product_id_l1c, product_id_l1b)

        # Verify attributes were set
        self.assertEqual(mock_ds.attrs["L1C_product_version"], product_id_l1c)
        self.assertEqual(mock_ds.attrs["L1B_product_version"], product_id_l1b)
        self.assertIn("generation_date", mock_ds.attrs)

        # Verify the save function was called
        mock_ds.to_netcdf.assert_called_once_with(l1c_path)
        mock_ds.close.assert_called_once()

    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.enrich_onesubswath_l1b")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.save_l1c_to_netcdf")
    @patch(
        "slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.os.path.exists",
        return_value=True,
    )
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.get_l1c_filepath")
    def test_main_logic_skips_if_overwrite_is_false(
        self, mock_get_path, mock_exists, mock_save, mock_enrich
    ):
        """
        Test that processing is skipped if the output file exists and overwrite is False.
        """
        mock_get_path.return_value = ("/fake/output.nc", "A01")

        cpt = l1c_script.do_l1c_safe_from_l1b_safe(
            full_safe_file="/fake/input.nc",
            productid="B49",
            outputdir="/tmp",
            product_configuration=self.fake_product_config,
            overwrite=False,
        )

        self.assertEqual(cpt["file_successfuly_written"], 0)
        self.assertEqual(cpt["output_file_already_present"], 1)
        mock_enrich.assert_not_called()
        mock_save.assert_not_called()

    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.glob")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.resource_strftime")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.get_start_date_from_attrs")
    def test_append_ancillary_field_file_not_found(
        self, mock_get_date, mock_strftime, mock_glob
    ):
        """
        Test that ancillary appending gracefully does nothing if the ancillary file is not found.
        """
        mock_get_date.return_value = datetime(2023, 1, 1)
        mock_strftime.return_value = (
            datetime(2023, 1, 1),
            "/path/to/missing_ancillary_*.nc",
        )
        mock_glob.return_value = []  # Simulate file not found

        ancillary_config = {"name": "ecmwf_0100_1h", "pattern": "pattern", "step": 1}

        ds_out, found, added = l1c_script.append_ancillary_field(
            ancillary_config, self.fake_ds
        )

        self.assertIs(ds_out, self.fake_ds)  # Should return the original dataset
        self.assertFalse(found)
        self.assertFalse(added)

    @patch(
        "slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.coloc_tiles_from_l1bgroup_with_raster"
    )
    @patch(
        "slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.raster_cropping_in_polygon_bounding_box"
    )
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.ecmwf_0100_1h")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.glob")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.resource_strftime")
    @patch("slcl1butils.scripts.do_WV_L1C_SAFE_from_L1B_SAFE.get_start_date_from_attrs")
    def test_append_ancillary_field_success_flow(
        self,
        mock_get_date,
        mock_strftime,
        mock_glob,
        mock_ecmwf_reader,
        mock_crop,
        mock_coloc,
    ):
        """
        Test the successful workflow for appending an ancillary field.
        """
        # Arrange
        mock_get_date.return_value = datetime(2023, 1, 1)
        mock_strftime.return_value = (
            datetime(2023, 1, 1),
            "/path/to/found_ancillary.nc",
        )
        mock_glob.return_value = ["/path/to/found_ancillary.nc"]  # Simulate file found

        fake_raster_ds = xr.Dataset({"wind_speed": [10]})
        mock_ecmwf_reader.return_value = fake_raster_ds
        mock_crop.return_value = fake_raster_ds
        mock_coloc.return_value = xr.Dataset({"coloc_wind": [10.5]})

        ancillary_config = {"name": "ecmwf_0100_1h", "pattern": "pattern", "step": 1}

        # Act
        ds_out, found, added = l1c_script.append_ancillary_field(
            ancillary_config, self.fake_ds_with_corners
        )

        # Assert
        self.assertTrue(found)
        self.assertTrue(added)
        # self.assertIn("ecmwf_0100_1h", ds_out.data_vars)  # Check that the merged variable is present
        self.assertIn(
            "coloc_wind", ds_out.coords
        )  # Check that the merged variable is present
        # self.assertEqual("coloc_wind",ds_out.name)
        mock_ecmwf_reader.assert_called_once()
        mock_crop.assert_called_once_with(
            ANY, fake_raster_ds
        )  # ANY for the shapely.Polygon
        mock_coloc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
