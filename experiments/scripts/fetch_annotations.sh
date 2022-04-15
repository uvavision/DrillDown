#!/bin/bash

scene_graph_xmls=https://www.cs.rice.edu/~vo9/fuwen/drilldown/sg_xmls.zip
region_graph_jsons=https://www.cs.rice.edu/~vo9/fuwen/drilldown/rg_jsons.zip

mkdir data/vg

echo "Downloading scene graph annotations"
wget $scene_graph_xmls -O sg_xmls.zip
echo "Unzipping..."
unzip -q sg_xmls.zip -d data/vg/

echo "Downloading region graph annotations"
wget $region_graph_jsons -O rg_jsons.zip
echo "Unzipping..."
unzip -q rg_jsons.zip -d data/vg/

rm sg_xmls.zip rg_jsons.zip
