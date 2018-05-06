#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate table of layer attributes for Nath & Thingbaijam (2012)
"""
import pandas as pd

LAYER_DEPTHS_KM = [0., 25., 70., 180., 300.]

LAYERS_DF = pd.DataFrame(
    list(zip(LAYER_DEPTHS_KM[:-1],
             LAYER_DEPTHS_KM[1:])),
    index=[1, 2, 3, 4],
    columns=['zmin', 'zmax'])
LAYERS_DF.index.name = 'layerid'
LAYERS_DF.to_csv('layers.csv')
print(LAYERS_DF)
