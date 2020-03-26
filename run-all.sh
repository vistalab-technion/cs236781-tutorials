#!/bin/sh

set -x

# Runs all the tutotial notebooks
jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.timeout=300 ./t*/*.ipynb
