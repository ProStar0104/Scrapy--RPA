#!/bin/bash
rm -f fs/all.*
nohup scrapy crawl fs -a parser=all -o fs/all.json > fs/all.out &
