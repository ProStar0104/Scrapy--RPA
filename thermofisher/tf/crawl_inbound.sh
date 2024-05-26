#!/bin/bash
rm -f inbound.*
nohup scrapy crawl tf -a parser=inbound -a target=https://www.thermofisher.com/search/se/contract-lab/us/en -o inbound.json >inbound.out &
