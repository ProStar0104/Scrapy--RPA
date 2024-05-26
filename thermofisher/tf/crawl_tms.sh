rm -f tf/tms.*
nohup scrapy crawl tf -a parser=tms -o tf/tms.json > tf/tms.out &
