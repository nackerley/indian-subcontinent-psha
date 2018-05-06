# Indian Subcontinent PSHA 

This is an implementation in [OpenQuake](https://github.com/gem/oq-engine)
of the probabilistic seismic hazard analysis (PSHA) model of
[Nath & Thingbaijam (2012)](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/83/1/135/143990).

* [Installation](#installation)
* [Usage](#usage)
* [Jobs](#jobs)
* [License](#license)
* [Contact](#contact)
* [Thanks](#thanks)

## Installation

The model is compatible with OpenQuake 2.6 and above.

Smoothed-gridded model files in Natural hazards’ Risk Markup Language (NRML)
format are too big to fit in a github repository but they
can be regenerated from smaller input data files and scripts, all included in
the repository.

### Clone repository

```bash
cd ~/src
git clone https://github.com/nackerley/indian-subcontinent-psha.git
```

### Install Anaconda

The OpenQuake NRML .xml model files are generated from .csv (source
model) and .tsv (logic tree) files using scripts that depend on the OpenQuake
Hazard Modeler's Toolikt (HMTK). This is done using python .py and Jupyter
Notebook scripts in .ipynb format.

The recommended way to install Jupyter
Notebook is to
[install Anaconda](https://docs.continuum.io/anaconda/install#linux-install).

### Create environment

After installing Anaconda, create an appropriate python environment to generate
the model files by running:

```bash
cd ~/src/indian-subcontinent-psha
conda env create --file oq.yml
source activate oq
```

This environment sets up OpenQuake with python 3. This way of installing
OpenQuake is currentlly in development; if you have any trouble you may wish to
use the environement defined by `oq2.yml` which calls for python 2 instead.

The `(oq)` kernel needs to be made accessible to jupyter in the root 
environment by running the following:
```bash
source activate oq
python -m ipykernel install --user --name oq
```
### Add utilities to Python path

Finally you must add the `~/src/indian-subcontinent-psha/utilities` directory
\to your PYTHONPATH. One way is to add the following to your `~\.bashrc`
```bash
export PYTHONPATH=$PYTHONPATH:~/src/indian-subcontinent-psha/utilities
```

### Verify installation

After installation you should be able to:
```python
import os
import sys
import source_model_tools as smt
```

## Usage

### Regenerate logic trees

To regenerate the logic tree XML, in your root environment run:
```bash
jupyter notebook
```

Open the following files in `~/src/indian-subcontinent-psha`
and `Cell ... Run All`:
1. `"Logic Trees/logic_trees_nt2012.ipynb"`

Note that the logic tree model XML files are small, and already included in
the repository, so this is only necessary if changes are made to
logic tree TSV description files.

### Regenerate source models

To regenerate the source XML, in your root environment run:
```bash
cd ~\Source Models
python write_source_models_nt2012.py
```
The smooothed-gridded models in particualr are large, so only "thinned"
versions are included in the repository for quality control purposes. Full
source models must be regenerated using the above script if the model si to
be used.

### Investigate models

The following jupyter notebooks are useful for investigating and visualizing
the models:

* `"Source Models/areal_source_models_nt2012.ipynb"`
* `"Source Models/smoothed_source_models_nt2012.ipynb"`
* `"Source Models/collapse_source_models.ipynb"`

### Ground motion predictino equation (GMPE) development

A set of jupyter notebooks in `~/src/indian-subcontinent-psha/GMPEs` were
used to generate validation data for the following modules of 
[openquake.hazardlib.gsim](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html):
* [gupta_2010](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.gupta_2010)
* [kanno_2006](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.kanno_2006)
* [nath_2012](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.nath_2012)
* [raghukanth_iyengar_2007](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.raghukanth_iyengar_2007)
* [sharma_2009](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#module-openquake.hazardlib.gsim.sharma_2009)
* [atkinson_boore_2003.AtkinsonBoore2003SSlabJapan](https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html#openquake.hazardlib.gsim.atkinson_boore_2003.AtkinsonBoore2003SSlabJapan)

## Jobs

Job configuration `job.ini` files are created manually rather than
automatically generated. Symbolic links to data files are include in each
folder, to avoid duplication of files.

The key job variants supported are:

* `"Jobs/cities_collapsed_v0"`
* `"Jobs/cities_collapsed_v1"`
* `"Jobs/cities_full_enumeration_v1"`
* `"Jobs/map_collapsed_v1"`

Note:
* `collapsed` vs. `full_enumeration` refers to whether or not
frequency-magnitude distributions are collapsed prior to hazard calculation.
`collapsed` source models will give correct results for mean hazard but not
for hazard quantiles or deaggregation.
* `cities` jobs are site-specific analyses, for the 18 cities listed
in Table 3 of Nath & Thingbaijam (2012).
* `map` refers to the 0.2° grid of 8102 points used to generate the data in the
Figure 7 and the electronic supplement of Nath & Thingbaijam (2012). 

## License

The **Indian Subcontinent PSHA model** is released under the
**[GNU Affero Public License 3](LICENSE.md)**.

## Contact

* Email: ackerley.nick@gmail.com

## Thanks

* [Graeme Weatherill](https://github.com/g-weatherill), **[GeoForschungsZentrum (GFZ) Potsdam](https://www.gfz-potsdam.de/)**
* [Marco Pagani](https://github.com/mmpagani), **[Global Earthquake Model Foundation (GEM)](http://gem.foundation)**
* [Kiran Kumar Singh Thingbaijam](https://ces.kaust.edu.sa/Pages/Kiran-Thingbaijam.aspx), **[King Abdullah University of Science & Technology (KAUST)](https://ces.kaust.edu.sa/)**
