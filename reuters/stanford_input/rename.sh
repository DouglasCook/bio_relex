#!/bin/bash
# needs to be run from file containing the NE tagged files

for file in *.out
do
    mv "$file" "`basename $file .txt.out`.txt"
done
