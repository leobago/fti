#!/bin/bash

for file in `ls results/*.dat`
do
    fn=`echo $file | cut -d '.' -f 1`
    export fname=$fn
    com="gnuplot vplot.plg"
    $com
done

