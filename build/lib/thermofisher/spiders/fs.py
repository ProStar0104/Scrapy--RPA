import re
from typing import Any

import pandas as pd
from scrapy.http import HtmlResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
from urlpath import URL

from .base import BaseSpider


class FishersciSpider(BaseSpider):
    debug = False
    name = 'fs'
    start_urls = ['https://www.fishersci.com']
    base_url = URL('https://www.fishersci.com')
    base_us_url = base_url / 'us/en'
    target_parser = None

    fs = Rule(
        link_extractor=LinkExtractor(
            allow_domains=['fishersci.com'],
            deny_domains=[
                'beta.fishersci.com',
                'de.fishersci.com',
                'eu.fishersci.com',
                'fr.fishersci.com',
                'info.fishersci.com',
                'labessentials.fishersci.com',
                'literature.fishersci.com',
                'punchout.fishersci.com',
                'supplierexchange.fishersci.com'
            ],
            deny=(
                '/_jcr_content/',
                '/antibody/product/',
                '/browse/',
                '/catalog/',
                '/forms-dev/',
                '/forms/',
                '/genome-database/',
                '/offers/',
                '/order/',
                '/programs',
                '/search/',
                '/shop/products/',
                '/store/',
                '/store1/',
                '/supportResources/')
        ),
        follow=True
    )
    """
    browse: PCP
    shop: PDP
    catalog: FCs
    """

    vaild_directories = [
        'amino-acids-reference-tool', 'brands', 'contactus', 'customer-help-support',
        'education-products', 'error', 'footer', 'healthcare-products', 'home',
        'our-response-to-the-covid-19-outbreak', 'periodic-table', 'products', 'programs',
        'science-social-hub', 'scientific-products'
    ]
    re_contentid = r'[A-Z\d]{8}'
    content_id_map = None

    def load_content_ids(self) -> None:
        df = pd.read_excel(self.base_dir / 'content ids.xlsx')
        # Only specific L1 and Active pages
        df = df[(df['l1_directory'].isin(self.vaild_directories)) & (df['cq:lastReplicationAction'] == 'Activate')].dropna(subset=['contentID'])
        df = df[['jcr:path', 'contentID']]
        df.columns = ['path', 'content id']
        df = df[df['content id'].str.contains(self.re_contentid, regex=True) == True]
        df['path'] = df['path'].str.replace('/content/fishersci/en_US/', '', regex=False)
        self.content_id_map = df
        return

    parser_map = {
        'all': {'rules': [fs], 'callback': 'parse_all'},
        'products': {'rules': [fs], 'callback': 'parse_products'},
        'fc': {'rules': [fs], 'callback': 'parse_fc'},
        'product_links': {'rules': [fs], 'callback': 'parse_product_links'}
    }

    products_parent_xpath = '//*[@id="content_tab"]/div/div/div/div[1]'
    main_parent_xpath = '//*[@id="main"]'

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.load_content_ids()

        if hasattr(self, 'parser'):
            if self.parser in self.parser_map:
                parser = self.parser_map[self.parser]
                self.rules = parser['rules']
                if 'callback' in parser:
                    for rule in self.rules:
                        # Assign the callback per parser_map only where one isn't already defined in the Rule
                        if not rule.callback:
                            rule.callback = parser['callback']
                            rule.errback = 'error'

                assert all(hasattr(self, rule.callback) for rule in self.rules), 'Invalid callback'
                if 'before_start' in parser:
                    parser['before_start'](self)

                if hasattr(self, 'target_url'):
                    self.logger.info(f'Targeting {self.target_url}')

            else:
                raise AttributeError(
                    f'Unrecognized parser {self.parser}. '
                    f'Valid options: {list(self.parser_map.keys())}')
        else:
            raise AttributeError(
                f'Must specify parser. Valid options: {list(self.parser_map.keys())}')

        self._compile_rules()  # Normally done in super().__init__() but we set self.rules after that call because we needed the kwargs attributes, so they were empty at the time. Call again now.
        self.logger.info(f'Parsers : {[rule.callback for rule in self.rules]}')

        if hasattr(self, 'target_url'):
            self.start_urls = [str(self.target_url)]
        else:
            with open(getattr(self, 'uris', self.base_dir / 'start_uris.txt'), 'r') as f:  # Use uris param or default to aem_uris.txt
                self.start_urls = [url[:-1] for url in f.readlines()]
                if not self.isdebugging():  # Help to troubleshoot only specified start URLs
                    self.start_urls.append('https://www.fishersci.com')
        return

    def check_host(self, host) -> URL:
        host = super().check_host(host)
        if host:
            parts = list(host.parts)
            if re.match(self.re_contentid, parts[-2]):
                content_id = parts[-2]
                directory = parts[-3]
                if directory in self.vaild_directories:
                    content_id_df = self.content_id_map[self.content_id_map['content id'].isin([content_id])]
                    paths = list(content_id_df['path'].values)
                    if len(paths) > 1:
                        self.logger.warning(f'Multiple paths found for content id {content_id}: {paths}')
                    elif len(paths) == 0:
                        self.logger.warning(f'No path found for content ID: {content_id}')
                    else:
                        host = (self.base_us_url / paths[0]).with_suffix('.html')
        return host

    def parse_products(self, response: HtmlResponse) -> dict:
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        parse_res = {}
        product_category_columns = sel.xpath(self.products_parent_xpath). \
            xpath(".//div[has-class('category_column')]")
        for col in product_category_columns:
            product_categories = col.xpath(".//div[has-class('alphabetical_order_category_list')]")
            for category in product_categories:
                category_res = []
                category_title = category.xpath('normalize-space(.//div/a/h4/span[1]/text())').get()
                category_prods = category.xpath('.//div/div/ul/li')
                for prod in category_prods:
                    prod_name = prod.css('a::text').get()
                    category_res.append(prod_name)
                parse_res[category_title] = category_res
        return parse_res

    def parse_fc(self, response: HtmlResponse) -> Any:
        # TODO: https://www.fishersci.com/us/en/scientific-products/special-offers-and-programs/limited-time-savings/consumables/jcr:content.html (embedded)

        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        fc_urls = self.normalize_links(sel.xpath('//*[@id="main"] | //*[@id="mainContent"]')
                                       .xpath('.//a[contains(@href,"/catalog/featured/")]/@href').getall(), host)
        fc_ids = set(re.search(r'/featured/(\d+)', url).group(1) for url in fc_urls)

        # Get embedded FCs
        efc_ids = set()
        embedded_fc = sel.xpath('.//meta[@name="tags"]/@content').get()
        if embedded_fc:
            for efc in [efc for efc in embedded_fc.split(',') if 'featured-collections' in efc]:
                efc_ids.add(re.search(r'/(\d+)$', efc).group(1))

        if fc_ids:
            yield {str(host): {'fc': fc_ids}}
        if efc_ids:
            yield {str(host): {'efc': efc_ids}}
        else:
            return

    def parse_product_links(self, response):
        """
        Collect information about product links
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)
        links = self.normalize_links(sel.xpath('//*[@id="main"] | //*[@id="mainContent"]').xpath('.//a[contains(@href,"/us/en/products/")]/@href'), host)
        if links:
            yield {str(host): links}

    def parse_all(self, response: HtmlResponse) -> dict:
        """
        Survey all pages found by the crawler
        """
        host = self.check_host(response)
        if host:
            yield {str(host): None}
        else:
            return None
