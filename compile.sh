#   !/bin/bash

# Determine if the OS is macOS
if [ "$(uname)" = "Darwin" ]; then
    CCL_DIR="ccl_cpu"
else
    # Check for the --cpu flag
    if [ "$1" = "--cpu" ]; then
        CCL_DIR="ccl_cpu"
    else
        CCL_DIR="ccl"
    fi
fi

# Build the components
cd ./models/post_processing/pa/
python setup.py build_ext --inplace
cd ../pse/
python setup.py build_ext --inplace
cd ../$CCL_DIR/
python setup.py build_ext --inplace
cd ../../../
