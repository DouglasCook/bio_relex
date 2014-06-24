#!/bin/bash

for file in *.out; do
    mv "$file" "`basename $file .txt.out`.txt"
done
