# NIST-DSE

This is the code to run algorithm for team Shawn in the NIST-DSE Competition. See more details here: [https://www.ecodse.org/](https://www.ecodse.org/).

Requied packages:
- scipy
- skimage
- numpy
- pandas
- matplotlib
- rasterio
- fiona

Steps:
- Download the raw data here [https://doi.org/10.5281/zenodo.867646](https://doi.org/10.5281/zenodo.867646).
- Set the appropriate data directory in `config.py`.
- run `python main.py` to fit the model and produce prediction polygons in the `Task1/predictions/` folder
