#!/bin/bash
rm -f tf/table_titles.*
nohup scrapy crawl tf -a parser=table_titles -o tf/table_titles.json > tf/table_titles.out &
