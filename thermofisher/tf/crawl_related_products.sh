rm -f related_products.*
nohup scrapy crawl tf -a parser=related_products -o related_products.json >related_products.out &

