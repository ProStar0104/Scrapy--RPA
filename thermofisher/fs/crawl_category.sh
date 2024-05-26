#!/bin/bash
rm -f fs/fs_category.*
nohup scrapy crawl fs -a parser=products -o fs/fs_category.json > fs/fs_category.out &
