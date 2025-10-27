#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/Main.py

# close the environment
deactivate