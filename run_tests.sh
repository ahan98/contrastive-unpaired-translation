#!/bin/sh
for file in tests/test_*.py; do
	echo "********* Running $file *********"
	python3 "$file" 
done
