#!/bin/bash
rm -f fs/product_links.*
nohup scrapy crawl fs -a parser=product_links -o fs/product_links.json > fs/product_links.out &
