import numpy as np


def ndvi_from_hs(i):
    red_bands = np.array([51,52,53,54,55,56,57,58,59])
    nir_bands = np.array([95,96,97,98,99,100])

    i_red = i[red_bands,:,:].mean(axis=0)
    i_nir = i[nir_bands,:,:].mean(axis=0)
    ndvi = (i_nir - i_red) / (i_nir + i_red)

    return ndvi