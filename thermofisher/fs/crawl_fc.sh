#!/bin/bash
rm -f fs/fc.*
nohup scrapy crawl fs -a parser=fc -o fs/fc.json > fs/fc.out &
