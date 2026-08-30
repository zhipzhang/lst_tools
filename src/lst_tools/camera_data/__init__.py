import os

from ctapipe.io import read_table

camera_data_path = os.path.join(os.path.dirname(__file__), "camera_description.h5")
camera_configuration = read_table(camera_data_path, "camera_geometry")

pix_x = camera_configuration["pix_x"]
pix_y = camera_configuration["pix_y"]
pix_area = camera_configuration["pix_area"]

__all__ = ["pix_x", "pix_y", "pix_area"]
