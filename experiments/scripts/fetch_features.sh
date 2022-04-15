#!/bin/bash

image_features=https://www.cs.rice.edu/~vo9/fuwen/drilldown/global_features.zip 
region_features=https://www.cs.rice.edu/~vo9/fuwen/drilldown/region_36_final.zip

mkdir data/vg

echo "Downloading image features"
wget $image_features -O global_features.zip
echo "Unzipping..."
unzip -q global_features.zip -d data/vg/

echo "Downloading region features"
wget $region_features -O region_36_final.zip
echo "Unzipping..."
unzip -q region_36_final.zip -d data/vg/

rm global_features.zip region_36_final.zip
