# Indian Subcontinent PSHA

This is an implementation in [OpenQuake](https://github.com/gem/oq-engine)
of the probabilistic seismic hazard analysis (PSHA) model of
[Nath & Thingbaijam (2012)](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/83/1/135/143990).

* [Installation](#installation)
* [Usage](#usage)
* [Jobs](#jobs)
* [Mapping](#mapping)
* [Data](#data)
* [License](#license)
* [Development](#development)
* [Contact](#contact)
* [Thanks](#thanks)

## Installation

Smoothed-gridded model files in Natural hazards’ Risk Markup Language (NRML)
format are too big to fit in a github repository but they
can be regenerated from smaller input data files and scripts, all included in
the repository.

The OpenQuake NRML .xml model files are generated from .csv (source
model) and .tsv (logic tree) files using scripts that depend on the OpenQuake
Hazard Modeler's Toolikt (HMTK). This is done using python .py and Jupyter
Notebook scripts in .ipynb format.

### Prerequisites

Ensure that the prerequisites are installed:
* OpenQuake (2.6 or newer): [oq-engine](https://github.com/gem/oq-engine/)
* Jupyter Notebook and python environment management:
[Anaconda](https://docs.continuum.io/anaconda/install#linux-install)
* lualatex (optional, for conversion of NRML logic trees to PDF):
[TexLive](https://tug.org/texlive/)

### Repository

Clone the repository:
```bash
cd ~/src
git clone https://github.com/nackerley/indian-subcontinent-psha.git
```

### Create environment

Create an environment with the necessary python packages:
```bash
cd ~/src/indian-subcontinent-psha
conda env create --file oq.yml
source activate oq
```

This environment sets up OpenQuake with python 3. This way of installing
OpenQuake is currentlly in development; if you have any trouble you may wish to
use the environement defined by `oq2.yml` which calls for python 2 instead.

Make the `(oq)` kernel accessible to jupyter in the root
environment by running the following:
```bash
source activate oq
python -m ipykernel install --user --name oq
```

To open jupyter notebooks which use this environment, run
the following in your root environment:
```bash
jupyter notebook
```
and select the appropriate files from the browser window which opens.

### Add utilities to Python path

Add `~/src/indian-subcontinent-psha/utilities` to your Python path.
One way is to append the following to your `~/.bashrc`:
```bash
export PYTHONPATH=$PYTHONPATH:~/src/indian-subcontinent-psha/utilities
```

### Verify installation

Verify your installation by trying:
```python
import source_model_tools as smt
```

## Usage

### Regenerate logic trees

To regenerate the logic tree XML, open and run the following jupyter notebooks:
* `"Logic Trees/logic_trees_nt2012.ipynb"`

Note that the logic tree model XML files are small, and already included in
the repository, so this is only necessary if changes are made to
logic tree TSV description files.

### Regenerate source models

To regenerate the source model XML, in your `(oq)` environment run:
```bash
cd ~/src/indian-subcontinent-psha/Source\ Models/
python write_source_models_nt2012.py
```
The smooothed-gridded models in particualr are large, so only "thinned"
versions are included in the repository for quality control purposes. Full
source models must be regenerated using the above script if the model is to
be used.

### Investigate source models

The following jupyter notebooks are useful for investigating and visualizing
the source models:

* `"Source Models/areal_source_models_nt2012.ipynb"`
* `"Source Models/smoothed_source_models_nt2012.ipynb"`
* `"Source Models/collapsed_source_models_nt2012.ipynb"`

## Jobs

Job configuration `job.ini` files are created manually rather than
automatically generated. Symbolic links to data files are include in each
folder, to avoid duplication of files.

The key job variants supported are:

* `Jobs/cities_collapsed_v0/`
* `Jobs/cities_collapsed_v1/`
* `Jobs/cities_full_disaggregation_v1/`
* `Jobs/map_collapsed_v1/`

Note:
* `collapsed` vs. `full` refers to whether or not frequency-magnitude
distributions were collapsed prior to hazard calculation.
`collapsed` source models will give correct results for mean hazard but not
for hazard quantiles or deaggregation.
* `cities` jobs are site-specific analyses, for the 18 cities listed
in Table 3 of Nath & Thingbaijam (2012).
* `map` refers to the 0.2° grid of 8102 points used to generate the data in the
Figure 7 and the electronic supplement of Nath & Thingbaijam (2012).

## Mapping

Files for generating maps using QGIS are stored at `Maps/`.
Maps incorporate data which can be downloaded separately from:
* **[Natural Earth](http://www.naturalearthdata.com)**
* HimaTibetMap-1.0: [Styron, Taylor & Okoronkwo (2010)](https://github.com/HimaTibetMap/HimaTibetMap)
* SLAB 1.0: [Hayes, Wald & Johnson (2012)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JB008524)
* Himalayan frontal thrust: [Berryman, Ries & Litchfield (2014)](http://www.nexus.globalquakemodel.org/gem-faulted-earth/)

## Data

This project draws heavily on published work. Portions of some of these
publications are included in this repository.

### Hazard model development

This model incorporates data from the following publications:
* `Data/nath2012probabilistic/`:
tables, digitized hazard curves and electronic supplement of
[Nath & Thingbaijam (2012)](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/83/1/135/143990)
* `Data/nath2011peak/`:
tables of
[Nath & Thingbaijam (2011)](https://link.springer.com/article/10.1007/s10950-010-9224-5)
* `Data/thingbaijam2011seismogenic/`:
Table 1 from "Thingbaijam & Nath (2011) A Seismogenic Source Framework for the Indian subcontinent" (unpublished)
* `Catalogue/` for model validation: 
[Nath, Thingbaijam & Ghosh (2010) Earthquake catalogue of South Asia - a generic MW scale framework](http://www.earthqhaz.net/sacat)

### Ground motion prediction equation (GMPE) development

A set of jupyter notebooks in `~/src/indian-subcontinent-psha/GMPEs` were
used to generate validation data for the following modules of
[openquake.hazardlib.gsim](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html):
* `GMPEs/gupta2010response/`:
[openquake.hazardlib.gsim.gupta_2010](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.gupta_2010)
* `GMPEs/kanno2006new/`:
[openquake.hazardlib.gsim.kanno_2006](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.kanno_2006)
* `GMPEs/nath2012ground/`:
[openquake.hazardlib.gsim.nath_2012](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.nath_2012)
* `GMPEs/raghukanth2007estimation/`:
[openquake.hazardlib.gsim.raghukanth_iyengar_2007](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.raghukanth_iyengar_2007)
* `GMPEs/sharma2009ground/`:
[openquake.hazardlib.gsim.sharma_2009](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.sharma_2009)
* `GMPEs/atkinson2003empirical/`:
[openquake.hazardlib.gsim.atkinson_boore_2003.AtkinsonBoore2003SSlabJapan](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#openquake.hazardlib.gsim.atkinson_boore_2003.AtkinsonBoore2003SSlabJapan)

## Development

Pull requests are welcome (though not expected, given the nature of the repository).

Useful tools for development:
* [nbdime](https://github.com/jupyter/nbdime)

## License

The **Indian Subcontinent PSHA model** is released under the
**[GNU Affero Public License 3](LICENSE.md)**.

## Contact

* Email: ackerley.nick@gmail.com

## Thanks

* [Graeme Weatherill](https://github.com/g-weatherill), **[GeoForschungsZentrum (GFZ) Potsdam](https://www.gfz-potsdam.de/)**
* [Marco Pagani](https://github.com/mmpagani), **[Global Earthquake Model Foundation (GEM)](http://gem.foundation)**
* [Kiran Kumar Singh Thingbaijam](https://ces.kaust.edu.sa/Pages/Kiran-Thingbaijam.aspx), **[King Abdullah University of Science & Technology (KAUST)](https://ces.kaust.edu.sa/)**
