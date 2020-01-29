#!/bin/bash

image1=https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
image2=https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

mkdir data/vg

echo "Downloading VG image1.zip ..."
wget $image1 -O image1.zip
echo "Unzipping..."
unzip -q image1.zip -d data/vg/

echo "Downloading VG image1.zip ..."
wget $image2 -O image2.zip
echo "Unzipping..."
unzip -q image2.zip -d data/vg

rm image1.zip image2.zip
