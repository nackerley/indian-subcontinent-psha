#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Indian Subcontinent PSHA
# Copyright (C) 2016-2018 Nick Ackerley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Select sites assuming models are well-defined out to extent of layer 1
of areal models, buffered in to the distance considered in hazard analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads

IN_BUFFER_M = -200e3
OUT_BUFFER_M = 100e3
WGS84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
INDIA_LCC_NSF = (
    '+proj=lcc +lat_1=12.472955 +lat_2=35.17280444444444 +lat_0=24 +lon_0=80 '
    '+x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs')

areal_df = pd.read_csv('../Source Models/nt2012_areal_source_model_v1.csv')
areal_df['geometry'] = areal_df['geometry'].apply(loads)
areal_gdf = gpd.GeoDataFrame(areal_df[areal_df['layerid'] == 1],
                             crs=WGS84)
region = gpd.GeoSeries(areal_gdf.geometry.unary_union, name='geometry',
                       crs=WGS84)
region_gdf = gpd.GeoDataFrame(region, crs=WGS84)
shrunk = (region_gdf.to_crs(INDIA_LCC_NSF)
          .buffer(OUT_BUFFER_M)
          .buffer(IN_BUFFER_M - OUT_BUFFER_M)
          .to_crs(WGS84))
shrunk.name = 'geometry'
shrunk_gdf = gpd.GeoDataFrame(shrunk, crs=WGS84)

lons, lats = np.meshgrid(np.arange(50, 110, 0.2), np.arange(-10, 50, 0.2))
sites_df = pd.DataFrame(np.vstack((lons.flatten(), lats.flatten())).T,
                        columns=['lons', 'lats'])
sites_df['coords'] = list(zip(sites_df['lons'], sites_df['lats']))
sites_df['geometry'] = sites_df['coords'].apply(Point)
sites_gdf = gpd.GeoDataFrame(sites_df[['geometry']], crs=WGS84)


sites = gpd.sjoin(shrunk_gdf, sites_gdf, how='right', op='contains')
sites.dropna(subset=['index_left'], inplace=True)
sites_df = pd.DataFrame(np.vstack((sites.geometry.x, sites.geometry.y)).T,
                        columns=['lons', 'lats'])
sites_df.sort_values(['lons', 'lats'], inplace=True)
sites_df.to_csv('extended_map.csv', index=False, header=False,
                float_format='%.1f')
