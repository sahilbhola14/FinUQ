#!/bin/bash
# Script to install the latest version of finuq
# NOTE: Change the shell Accodingly
# Begin User Inputs
module_folder="modules"
# End User Inputs

module_path=$(realpath "$module_folder")

if [[ $1 == "--uninstall" ]] || [[ $1 == "-u" ]]; then
    # Uninstall finuq
    echo "Uninstalling finuq"
    pip uninstall finuq

elif [[ $1 == "--install" ]] || [[ $1 == "-i" ]]; then
    # Install finuq
    echo "Installing finuq"
    pip install -e $module_path
fi
