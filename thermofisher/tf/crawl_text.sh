rm -f tf/text.*
nohup scrapy crawl tf -a parser=text -o tf/text.json > tf/text.out &
