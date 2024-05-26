rm -f headings.*
nohup scrapy crawl tf -a parser=heading_type -o headings.json >headings.out &
