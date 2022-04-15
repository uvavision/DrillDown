#!/bin/bash

PRTRAINED_URL=https://www.cs.rice.edu/~vo9/fuwen/drilldown/pretrained.zip

mkdir data/caches/

echo "Downloading pretrained model ..."
wget $PRTRAINED_URL -O pretrained.zip
echo "Unzipping..."
unzip -q pretrained.zip -d data/caches/

rm pretrained.zip
