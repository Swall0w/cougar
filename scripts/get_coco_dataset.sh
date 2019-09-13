#!/bin/bash

DATAPATH=~/dataset/coco
mkdir -p ${DATAPATH}
cd ${DATAPATH}

# Download Images
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
#wget -c http://images.cocodataset.org/zips/test2014.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#wget -c http://images.cocodataset.org/annotations/image_info_test2014.zip

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip
#unzip -q test2014.zip
unzip -q annotations_trainval2014.zip
#unzip -q image_info_test2014.zip
