
import geopandas
import pandas as pd
import rasterio
import csv
import numpy as np
import pyproj
from pyproj import Proj, transform, Transformer
import skimage


def load_qgis_gcps(filename, src_crs='epsg:4326', dst_crs='epsg:4326'):

    # Some GCP files use 'epsg:3857' as src_crs

    gcps = []
    with open(filename, 'r') as csv_file:

        # Open CSV file
        reader = csv.reader(csv_file)

        # Skip first two rows (header)
        next(reader, None)
        next(reader, None)

        # Create transformer
        transformer = Transformer.from_crs(src_crs, dst_crs)

        # Iterate through rows
        for gcp in reader:

            # Transform lat/lon
            if src_crs.lower() != dst_crs.lower():
                lon = gcp[0]
                lat = gcp[1]
                lon, lat = transformer.transform(lon, lat)
            else:
                # TODO: figure out why these need to be swapped
                lon = gcp[1]
                lat = gcp[0]

            # Read GCP fields from row
            gcps.append(rasterio.control.GroundControlPoint(row=float(gcp[2]),
                                                            col=float(gcp[3]), 
                                                            x=float(lon), 
                                                            y=float(lat)))

        # Close file
        csv_file.close()

    # Return list of GCPs
    return gcps



def calculate_poly_gcp_transforms_skimage(gcps):

    # https://scikit-image.org/docs/stable/api/skimage.transform.html#polynomialtransform

    img_coords = np.zeros((len(gcps),2))
    geo_coords = np.zeros((len(gcps),2))

    # Load image coords and geospatial coords from GCPs.
    for i,gcp in enumerate(gcps):
        img_coords[i,0] = gcp.row
        img_coords[i,1] = gcp.col
        geo_coords[i,0] = gcp.y
        geo_coords[i,1] = gcp.x
    
    # Estimate transform
    t = skimage.transform.estimate_transform('polynomial', img_coords, geo_coords, 2)

    # Get coefficients from transform
    lat_c = t.params[0]
    lon_c = t.params[1]

    # Return the coefficients
    return lat_c, lon_c



def calculate_poly_geo_coords_skimage(X, Y, lat_c, lon_c):
    
    #X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))

    #x.T = [a00 a10 a11 a20 a21 a22 ... ann
    #   b00 b10 b11 b20 b21 b22 ... bnn c3]

    #X = (( a_00 * x**(0 - 0) * y**0 ))
    #(( a_10 * x**(1 - 0) * y**0 ))  +  (( a_11 * x**(1 - 1) * y**1 ))
    #(( a_20 * x**(2 - 0) * y**0 ))  +  (( a_21 * x**(2 - 1) * y**1 )) 
    #                                +  (( a_22 * x**(2 - 2) * y**2 ))
   
    c = lat_c
    lat = c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2

    c = lon_c
    lon = c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2

    return (lat, lon)