# fair-transit-india
## Python Environment

```bash
cd ~ # or wherever you want to install the environment
apt-get install python3.11 python3.11-venv
# check you have python3.11 installed
python3.11 --version
# create a virtual environment and activate it, will create a directory env311 with the python enviornment in the current directory
python3.11 -m venv env311
source env311/bin/activate
# ensure that the current python environment is now 3.11
python --version
python -m pip install -r fair-transit-india/requirements.txt # or change path so it points to requirements.txt in this repository
```
## Network Creation
To create the underlying network (edge list), using zones and centroids

run notebook: 

```bash
./notebooks/generate_network.ipynb
```

OD matrices generation

run notebook:
```bash
./notebooks/generate_od.ipynb
```

## Run

Single Budget Run: 

```bash
# in src/config.py configure parameters for single a single budget run
python src/FIT.py [numerical_budget]
```