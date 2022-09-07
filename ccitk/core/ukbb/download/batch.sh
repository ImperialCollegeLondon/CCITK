#!/bin/bash

python master.py --csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv --key-path /vol/biodata/data/biobank/40616/key/ukbb.key --ukbfetch-path /vol/biodata/data/biobank/40616/utils/ukbfetch --output-dir /vol/biodata/data/biobank/40616/output --n-partition 10 --n-thread 0