rm -f orphans.*
nohup scrapy crawl tf -a parser=orphans -s ROBOTSTXT_OBEY=False -o orphans.json >orphans.out &
