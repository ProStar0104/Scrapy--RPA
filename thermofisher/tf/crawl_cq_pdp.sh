#!/bin/bash
rm -f tf/cq_pdp.*
nohup scrapy crawl tf -a parser=cq_pdp -o tf/cq_pdp.json > tf/cq_pdp.out &
