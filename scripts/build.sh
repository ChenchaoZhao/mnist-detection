#!/bin/bash

rm -r build
rm -r dist
rm -r mnist-detection.egg-info

python setup.py sdist bdist_wheel