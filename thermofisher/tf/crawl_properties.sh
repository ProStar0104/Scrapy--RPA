rm -f tf/properties.*
nohup scrapy crawl tf -a parser=properties -o tf/properties.json > tf/properties.out &
