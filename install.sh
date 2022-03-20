# Get code from GITHUB
rm -fr mpslib && git clone https://github.com/algocompretto/mpslib.git
cd mpslib
make

# install scikit-mps
cd scikit-mps
pip install .
