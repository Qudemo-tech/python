#!/usr/bin/env bash

# Install build requirements before running pip install
pip install --upgrade pip setuptools wheel

# Now install the app dependencies
pip install -r requirements.txt
