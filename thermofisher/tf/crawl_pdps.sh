rm -f pdps.*
nohup scrapy crawl tf -a parser=pdps -o pdps.json >pdps.out &
