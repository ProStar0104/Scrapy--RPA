#!/bin/bash
rm -f tf/videos.*
nohup scrapy crawl tf -a parser=videos -o tf/videos.json > tf/videos.out &
