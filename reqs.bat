python -m pip install pip --upgrade
python -m pip install pyarrow pyarrow-stubs
python -m pip install numpy
python -m pip install matplotlib
python -m pip install statsmodels
python -m pip install -e ./packages
python -m pip freeze > reqs.txt
