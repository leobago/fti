#!/bin/bash
rm -rf Global/* Local/* Meta/*
for i in $(ls *.fti)
do
    echo $i
    sed 's/.*failure.*/failure=0/' $i > tmp
    mv tmp  $i
done    
