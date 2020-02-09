#!/usr/bin/env bash

# script to download and unzip the training and validation data
# and put them in the right directory

unzip_from_link() {
  local link=$1
  local dir=$2

  echo $link
  echo $dir

  curl -o "${dir}/tmp.zip" $link
  echo "Unziping..."
  unzip -q "${dir}tmp" -d ${dir}
  echo "Removing tmp file"
  $(rm "${dir}tmp.zip")
}

echo "Created the Data directory"
$(mkdir -p "Data/")

echo "Downloading Training Data"
unzip_from_link "http://rpg.ifi.uzh.ch/datasets/sim2real_ddr/simulation_training_data.zip" "Data/"

echo "Downloading Validation Data"
unzip_from_link "http://rpg.ifi.uzh.ch/datasets/sim2real_ddr/validation_real_data.zip" "Data/"
