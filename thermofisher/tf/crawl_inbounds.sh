#!/bin/bash
rm -f inbound_*.*

nohup scrapy crawl tf -a parser=inbound -a target_url=www.unitylabservices.com -o inbound_u.json >inbound_u.out &
nohup scrapy crawl tf -a parser=inbound -a target_url=www.labequipmentparts.com -o inbound_l.json >inbound_l.out &
nohup scrapy crawl tf -a parser=inbound -a target_url=www.scientificinstrumentparts.com -o inbound_s.json >inbound_s.out &
nohup scrapy crawl tf -a parser=inbound -a target_text="unity lab services" -o inbound_uls.json >inbound_uls.out &
nohup scrapy crawl tf -a parser=inbound -a target_url=promotions.thermofisher.com -o inbound_pt.json >inbound_pt.out &
nohup scrapy crawl tf -a parser=inbound -a target_url=resource.thermofisher.com -o inbound_rt.json >inbound_rt.out &
nohup scrapy crawl tf -a parser=inbound -a target_url=kb.unitylabservices.com -o inbound_kb.json >inbound_kb.out &
