#!/bin/sh
for file in ./test_*.py; do
	echo "********* Running $file *********"
	python3 "$file" 
done
