#!/bin/bash

# Make sure the libraries are visible
libpath_set=`echo $LD_LIBRARY_PATH | grep modules/build/lib`
if [[ $libpath_set == "" ]]
then
    export LD_LIBRARY_PATH=`pwd`/modules/build/lib:$LD_LIBRARY_PATH
fi

# Rebuild the source code and run the application
make rebuild
./bin/main.out

