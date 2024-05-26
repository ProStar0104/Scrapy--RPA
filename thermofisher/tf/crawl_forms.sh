rm -f forms.*
nohup scrapy crawl tf -a parser=forms -o forms.json > forms.out &
