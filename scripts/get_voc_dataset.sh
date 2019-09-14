#!/bin/bash

DATAPATH=~/dataset/
mkdir -p ${DATAPATH}
cd ${DATAPATH}

# Download Images
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# Unzip
tar xf VOCtrainval_11-May-2012.tar && rm VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar && rm VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar && rm VOCtest_06-Nov-2007.tar
