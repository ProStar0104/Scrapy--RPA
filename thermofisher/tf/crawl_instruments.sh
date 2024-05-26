rm -f tools3p.*
nohup scrapy crawl tf -a parser=tools3p -o tools3p.json >tools3p.out &
