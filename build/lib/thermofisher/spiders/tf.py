import re
from concurrent.futures import as_completed
from copy import deepcopy
from datetime import datetime as dt
from distutils.util import strtobool
from math import ceil
from typing import Any, List, Union
from urllib.parse import unquote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from deprecation import deprecated
from scrapy import Selector
from scrapy.http import HtmlResponse, Request
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import SelectorList
from scrapy.spiders import Rule
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support.wait import WebDriverWait
from urlpath import URL

from .base import BaseSpider


class TfSpider(BaseSpider):
    debug = False
    name = 'tf'
    allowed_domains = ['thermofisher.com']

    # /browse/category = product categories
    # /order/catalog = PDPs

    css_main = 'div.main,div.container-fluid,div.cmp-p-text,div.cmp-p-container,div#homepage'
    css_left_nav = 'div.cmp-anchorlist,div.container-leftnav,div.leftnav'

    # Custom Groups / Featured Collections
    re_custom_group_names = re.compile(r'customGroup=(.+)')
    re_custom_group_ids = re.compile(r'/browse/featured/us/en/(\d+)/')

    re_orphans = [re.compile(r'thermofisher\.com/order/(?!catalog/product/)', re.IGNORECASE),
                  re.compile(r'(?<!www)\.thermofisher\.com', re.IGNORECASE),
                  re.compile(r'thermofisher\.com(?!/us/en)(?!.*(/product/))', re.IGNORECASE)]

    # Captured links
    ids_exclude = ['globalHeaderInclusion', 'meganav-content', 'meganav-extended']  # div ids to exclude from content
    pdplink_xpath = ".//a[contains(@href,'/order/catalog/product/') or (contains(@href,'adirect/lt?cmd=') and (contains(@href,'catProductDetail')))]"
    pcplink_xpath = ".//a[contains(@href,'/category/us/en/') or contains(@href,'/browse/category/') or (contains(@href,'adirect/lt?cmd=') and (contains(@href,'DisplayCategory')))]"
    fc_xpath = ".//a[contains(@href,'/search/browse/results?customGroup=') or contains(@href,'/search/browse/featured/')]/@href"  # Old and new format

    re_sku_href = re.compile(r'/order/catalog/product/|/browse/category/|(adirect/lt\?cmd=.*(catProductDetail|DisplayCategory))')
    deprecated_xpath = ".//a[contains(@href,'/adirect/lt?cmd=') and not (contains(@href,'catProductDetail') or contains(@href,'DisplayCategory'))]/@href"
    navid_xpath = ".//a[contains(@href,'/search/results?query')]"
    rackselector_xpath = ".//a[contains(@href,'rackselector.thermofisher')]"

    # Rules
    # Note: Rules apply only to the initial link. Redirects do not get put through these Rules (Scrapy limitation: https://github.com/scrapy/scrapy/issues/1744)
    #       So redirect targets that don't comply with the rules will still get crawled. Need secondary rule check: check_host()
    cq = Rule(link_extractor=LinkExtractor(allow_domains='www.thermofisher.com',
                                           allow=('/us/en/', '/wo/en/', '/cy/en/', '/uk/en/'),
                                           deny=('/search/', '/order/catalog/', '/antibody/', '/reference-components/', '/_jcr_content/', r'\?')),
              follow=True)

    forms = deepcopy(cq)
    # forms.link_extractor.deny_res.append(re.compile(r'/forms/', re.UNICODE))  # Exclude /forms/ directory to find forms elsewhere
    forms.link_extractor.allow_res.append(re.compile(r'/global/forms/', re.UNICODE))

    magellan = Rule(link_extractor=LinkExtractor(allow_domains='www.thermofisher.com',
                                                 allow=(),
                                                 deny=('/reference-components/', '/_jcr_content/')),
                    follow=True)

    faq = Rule(link_extractor=LinkExtractor(), follow=True)

    orphans = Rule(link_extractor=LinkExtractor(allow_domains='thermofisher.com',
                                                deny_domains=['preview.prod.thermofisher.com', 'prod-auth.thermofisher.com'],
                                                deny=('/search/',
                                                      '/search/browse/category/',
                                                      '/order/catalog/product/',
                                                      '/order/genome-database/',
                                                      '/elisa/',
                                                      '/blog/',
                                                      '/antibody/',
                                                      '/reference-components/', '/_jcr_content/', r'\?')),
                   follow=True)

    pdps = [Rule(link_extractor=LinkExtractor(allow=('/us/en/', '/search/'), deny=('/antibody/',)), callback='follow_ajax', follow=True),
            Rule(link_extractor=LinkExtractor(allow=('/order/catalog/product/',)), callback='parse_links', follow=True)]

    pdp_related_products = [Rule(link_extractor=LinkExtractor(allow=('/order/catalog/product/',)), callback='parse_related_products', follow=False)]

    def cq_pdp_start(self):
        self.target_types = {'pdp': self.pdplink_xpath,
                             'pcp': self.pcplink_xpath,
                             'fc': self.fc_xpath,
                             'primary hijack': self.detect_primary_hijacks,
                             'secondary hijack': self.detect_secondary_hijacks}
        self.load_pdp_hijack()
        return

    def load_trademarks(self) -> None:
        """
        Trademark:Trademark RE
        """

        def load_file(file, case_sensitive: bool):
            flags = re.IGNORECASE if not case_sensitive else 0
            with open(self.base_dir / file, 'r', encoding='utf-8') as f:
                tms = [trademark.strip() for trademark in f.readlines()]
                tms = [self.normalize_punctuation(tm) for tm in tms]
                tms = {trademark: re.compile(rf'\b({re.escape(trademark)})\b', flags=flags) for trademark in tms}
            return tms  # , self.chunk_dict(tms, len(tms) // self.n_cpus)

        tms = load_file('trademarks.txt', True)  # Case sensitive
        tms.update(load_file('trademarks_i.txt', False))  # Case insensitive
        self.tms = tms
        self.tm_chunks = self.chunk_dict(tms, ceil(len(tms) / self.mp.n_cpus))
        return

    def load_pdp_hijack(self):
        hijacks = pd.read_csv(self.base_dir / 'search_hijack.csv')
        hijacks.columns = ['alias', 'url']
        hijacks.url = hijacks.url.str.replace(r'^https?://', '', regex=True)
        hijacks.url = hijacks.url.map(re.escape)

        varmap = {'{daydomain}': 'www.thermofisher.com',
                  '{countrycode}': r'(\w{2}|global)',
                  '{langcode}': r'\w{2}',
                  '{clonesdomain}': 'clones.thermofisher.com',
                  '{orfdomain}': 'orf.thermofisher.com',
                  '{rnaidesignerdomain}': 'rnaidesigner.thermofisher.com',
                  }
        for var, val in varmap.items():
            hijacks.url = hijacks.url.str.replace(re.escape(var), val, regex=False)

        hijacks.url = hijacks.url.map(re.compile)
        self.hijacks = hijacks
        return

    # Parser Parameter (-a parser= ) mapping
    parser_map = {'cq_pdp': {'rules': [cq], 'callback': 'parse_cq_to_pdp', 'before start': cq_pdp_start},
                  'orphans': {'rules': [orphans], 'callback': 'parse_orphans'},
                  'pdps': {'rules': pdps, 'callback': None},
                  'tools3p': {'rules': [cq], 'callback': 'parse_tools3p'},  # Third party tools
                  'subtitles': {'rules': [cq], 'callback': 'parse_subtitles'},
                  'text': {'rules': [cq], 'callback': 'parse_text'},
                  'properties': {'rules': [cq], 'callback': 'parse_properties'},
                  'faq': {'rules': [cq], 'callback': 'parse_faq'},
                  'inbound': {'rules': [cq], 'callback': 'parse_inbound_links'},  # Additional params: target and exact (optional)
                  'videos': {'rules': [cq], 'callback': 'parse_videos'},
                  'heading_type': {'rules': [cq], 'callback': 'parse_heading_type'},
                  'related_products': {'rules': pdp_related_products},
                  'magellan': {'rules': [magellan], 'callback': 'parse_properties'},
                  'forms': {'rules': [forms], 'callback': 'parse_forms'},
                  'tms': {'rules': [cq], 'callback': 'parse_trademarks', 'before start': load_trademarks},
                  'table_titles': {'rules': [cq], 'callback': 'parse_table_titles'}
                  }

    def __init__(self, *args, **kwargs):
        """
        This crawler supports a dynamic parser argument, passed using the -a parameter for scrapy crawl
        The parameter is called 'parser' and valid options are the keys in self.parser_map.
        Example: scrapy crawl tf -a parser=cq_pdp
        """
        super().__init__(*args, **kwargs)  # kwargs are set as instance attributes inside here

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
                if 'before start' in parser:
                    parser['before start'](self)  # Call before start function

                if hasattr(self, 'target_url'):
                    self.target_url = self.normalize_links([self.target_url], host=URL(self.target_url), ignore_scheme=True, strip_suffix=False, as_url=True)[0]
                    self.logger.info(f'Targeting {self.target_url}')

            else:
                raise AttributeError(f'Unrecognized parser {self.parser}. Valid options: {list(self.parser_map.keys())}')
        else:
            raise AttributeError(f'Must specify parser. Valid options: {list(self.parser_map.keys())}')

        self._compile_rules()  # Normally done in super().__init__() but we set self.rules after that call because we needed the kwargs attributes, so they were empty at the time. Call again now.
        self.logger.info(f'Parsers : {[rule.callback for rule in self.rules]}')

        # # TODO: Temporary logic. Integrate with parser map
        # products = self.base_dir / 'products.csv'
        # if products.exists():
        #     products = pd.read_csv(products).squeeze()
        #     # products = 'https://www.thermofisher.com/order/catalog/product/' + products
        #     self.start_urls = products.tolist()
        # else:
        if hasattr(self, 'target_url'):
            self.start_urls = [str(self.target_url)]
        else:
            with open(getattr(self, 'uris', self.base_dir / 'aem_uris.txt'), 'r') as f:  # Use uris param or default to aem_uris.txt
                self.start_urls = [url[:-1] for url in f.readlines()]
                if not self.isdebugging():  # Help to troubleshoot only specified start URLs
                    self.start_urls.append('https://www.thermofisher.com')

        assert len(self._rules) > 0, 'No Rules defined'  # Empty rules can cause and be caused by very subtle bugs
        return

    # if self.debug:
    #     proxy = 'http://localhost:5555'
    #     with open('starturis.txt', 'r') as f:
    #         return [Request(url=url[:-1], callback=self.parse, meta={'proxy': proxy}) for url in f.readlines()]
    # else:
    #     # proxy = ''
    #     with open('starturis.txt', 'r') as f:
    #         return [Request(url=url[:-1], callback=self.parse) for url in f.readlines()]

    def error(self, failure):
        pass

    def follow_ajax(self, response):
        """
        Follow Ajax links
        """
        soup = BeautifulSoup(response.body, features='lxml')
        sel = Selector(text=soup.prettify())

        # Extract CQ pages with links to PDP
        # Example: https://www.thermofisher.com/us/en/home/life-science/protein-biology/protein-gel-electrophoresis/protein-gels/bolt-bis-tris-plus-gels.html
        host = URL(response.url)
        plc_products = self.plc_products(sel, host)
        if plc_products:
            self.logger.info(f'Found {len(plc_products):,} PLC products')
            for link in plc_products:
                yield response.follow(link, self.parse_links)

    def find_link_patterns(self, response):
        """
        Find pages containing certain link patterns
        """
        target = ".//a[contains(@href,'website-overview')]"  # Link pattern to look for

        host = URL(response.url).with_query('').with_fragment('')

        soup = BeautifulSoup(response.body, features='lxml')
        sel = Selector(text=str(soup))
        links = []

        for link in sel.css(self.css_main).xpath(target):
            soup = BeautifulSoup(link.get())
            links.append({'URL': soup.a['href'], 'Text': soup.a.text.strip()})

        if links:
            yield {str(host): links}

    def follow_hidden_links(self, selector, host):
        for link in self.normalize_links(selector.xpath('.//input[@type="hidden" and contains(@value,"/en") and contains(@value,".htm")]/@value'), host):
            self.logger.info(f'Following hidden link {link}')
            yield Request(link, callback=eval('self.' + self.rules[0].callback))

    def selector_products(self, selector_pages, host):
        # Follow selector page and collect SKUs from it
        selector_products = set()
        for selector_page in selector_pages:
            selector_response = requests.get(selector_page)
            soup = BeautifulSoup(selector_response.content, features='lxml')
            selector_sel = Selector(text=soup.prettify())
            # Collect Selector SKUs
            selector_products = selector_products.union(self.normalize_links(selector_sel.css(self.css_main).xpath(self.pdplink_xpath), host))

        return selector_products

    @deprecated(details='CG resolution now happens in EKG from Hybris')
    def custom_group_skus(self, custom_groups):
        """
        Given a list of custom group URLs, return their linked SKU PDP's
        Custom Groups are redirected to a newer format:
        Link Target:  https://www.thermofisher.com/search/browse/results?customGroup=Immunohistochemistry+Consumables+and+Reagents
        Redirects to: https://www.thermofisher.com/search/browse/featured/us/en/80012831/Immunohistochemistry+Consumables+And+Reagents
        We then add the query ?resultsPerPage=1000 to the redirected link to avoid pagination

        :param custom_groups: List of custom group URLs
        :return: Flat list of SKUs (not organized by original custom group)
        """
        skus = []
        throttle = self.get_throttle()
        for link in custom_groups:
            with throttle:
                resp = requests.get(link)  # This link redirects
                resp = requests.get(URL(resp.url).with_query(resultsPerPage=1000))  # Take redirected URL (resp.url) and add ?resultsPerPage to get all SKUs on one page
                self.logger.info(f'Expanding custom group: {link}')
                if resp.status_code == requests.codes.ok:
                    soup = BeautifulSoup(resp.text, features='lxml')
                    sku_list = self.normalize_links(Selector(text=str(soup)).css('div.result-container').xpath('.//h2/a/@href'), link)
                    if sku_list:
                        skus += sku_list
                        self.logger.info(f'Collected {len(sku_list):,} SKUs from custom group')
                    else:
                        self.logger.warning(f'No SKUs found for Custom Group {link}')
        return list(set(skus))

    def parse_forms(self, response):
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)
        content = sel.css(self.css_main)
        forms = content.xpath('.//form')
        for form in forms:
            yield {str(host): {
                'id': form.xpath('@id').get(),
                'name': form.xpath('@name').get(),
                'form-id': form.xpath('input[@name=":formid"]/@value').get() or form.xpath('input[@name="form-id"]/@value').get(),
                'gcmsFormId': form.xpath('input[@name="gcmsFormId"]/@value').get(),
                'elqSiteID': form.xpath('input[@name="elqSiteID"]/@value').get(),
                'elqFormName': form.xpath('input[@name="elqFormName"]/@value').get(),
                'redirect': form.xpath('input[@name=":redirect"]/@value').get()
            }}

    def parse_orphans(self, response):
        """
        Identify orphaned pages
        """
        host = self.check_host(response)
        if not host:
            return None
        host = str(host)

        if any([pat.search(host) for pat in self.re_orphans]):
            referer = response.request.headers.get('Referer', None)
            if referer:
                referer = referer.decode('utf-8')
            yield {host: {'referer': referer}}

    def plc_products(self, sel, host):
        """
        Collect product links from Product List Components
        """
        ajax_links = sel.css('div.productlist,div.cmp-p-productlist').xpath('.//*[contains(text(),"ajax")]').getall()

        plc_products = set()
        for ajax_link in ajax_links:
            ajax_link = self.re_ajax.search(ajax_link)
            if ajax_link:
                ajax_link = ajax_link.groups()[0]

                # Load each Ajax-based SKU list
                ajax_response = requests.get(str(host / ajax_link))
                soup = BeautifulSoup(ajax_response.content, features='lxml')
                ajax_sel = Selector(text=soup.prettify())
                # Collect SKUs
                plc_products = plc_products.union(self.get_products(ajax_sel, host))  # Add to products set
        return SelectorList(plc_products)

    @deprecated(details='Using Selenium now')
    def bespoke_products(self, sel, host):
        """
        Collect Bespoke products (using JSON list)
        Manually parse JS
        """
        sku_keys = ['catalog', 'sku']
        skus = []
        throttle = self.get_throttle()
        for script in sel.xpath('.//script[contains(text(),"json") and contains(text(),"request")]').getall():
            json_request = re.search(r'request.+json\((.+)\);', script, flags=re.IGNORECASE)
            if json_request:
                jsonpath = json_request.groups()[0]
                literal = re.match(r'"(.+)"', jsonpath)
                if literal:
                    # Extract literal from double quotes
                    jsonpath = literal.groups()[0]
                else:
                    # Resolve Variable
                    variable = re.search(rf'{jsonpath}\s*=\s*"(.+)"\s*;', script)
                    if variable:
                        jsonpath = variable.groups()[0]
                    else:
                        jsonpath = None
                # Request the path
                if jsonpath:
                    with throttle:
                        ret = requests.get(str(host / jsonpath))
                        if ret.status_code == requests.codes['ok']:
                            bespoke_json = ret.json()
                            skus.extend([item[sku_key] for sku_key in sku_keys for item in bespoke_json if sku_key in item])
        return skus

    @staticmethod
    def is_tfrefresh(selector):
        """
        Is the content a TF Refresh page?
        :param selector: Content to inspect
        :return: True if TF Refresh, else False
        """
        return len(selector.css('head>link').xpath('//*[contains(@href,"/content/dam/tfsite/selection-guides/")]')) > 0

    def detect_hijack(self, link, host):
        """
        Hijack Detection
        """

        hijacks = []
        for hijack in self.hijacks.itertuples():
            if hijack.url.search(link):
                hijacks.append(hijack.alias)
        return hijacks

    def detect_primary_hijacks(self, sel, host):
        """
        sel is unused but needed for signature uniformity with detect_secondary_hijacks
        :param sel:
        :param host:
        :return:
        """
        # Primary Hijacks (Page1 → Products)
        return self.detect_hijack(str(host), str(host))

    def detect_secondary_hijacks(self, sel, host):
        # Secondary Hijacks (Page2 → Page1 → Products → Associate Page2 with Products)
        pool = []
        for link in [link for link in self.normalize_links(sel.xpath('.//a/@href'), host) if link != str(host)]:
            pool.append(self.thread_executor.submit(self.detect_hijack, link, host))
        hijacks = []
        for future in as_completed(pool):
            for products in future.result():
                hijacks.append(products)

        return hijacks

    def get_products(self, selector: Union[Selector, SelectorList, List], host):
        if isinstance(selector, Selector) or isinstance(selector, SelectorList):
            # Select the whole table row
            skus = selector.xpath(self.pdplink_xpath).xpath('ancestor::tr[1]')
        elif isinstance(selector, List):
            skus = [link for link in selector if link and self.re_sku_href.search(link)]
        else:
            raise TypeError('selector must be Selector or List')
        return skus

    def collect_cg_products_by_id(self, sel, host):
        custom_groups = self.normalize_links(sel.xpath(self.fc_xpath), host, strip_query=False)
        custom_group_ids = [match.group(1) for match in [self.re_custom_group_ids.search(cg) for cg in custom_groups] if match]
        return custom_group_ids

    def collect_cg_products_by_name(self, sel, host):
        custom_groups = self.normalize_links(sel.xpath(self.fc_xpath), host, strip_query=False)
        custom_group_names = [unquote_plus(match.group(1)) for match in [self.re_custom_group_names.search(cg) for cg in custom_groups] if match]
        return custom_group_names

    def parse_videos(self, response):
        """
        Get pages containing Brightcove videos
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response, exclude_nav=True)

        # Content -headers -left nav
        content = sel.css(self.css_main) \
            .xpath('.//*[not(ancestor-or-self::div[@id="globalHeaderInclusion"])]') \
            .xpath('.//*[not(ancestor-or-self::div[@class="container-leftnav"])]') \
            .xpath('.//*[not(ancestor-or-self::div[@class="cmp-anchorlist"])]')

        video_ids = list(set(content.css('div.brightcove-container>video').xpath('./@data-video-id').getall()))
        modal_video_ids = list(set(content.css('div.brightcoveplayer>a>img').xpath('./@src').getall()))
        modal_video_ids = [re.search(r'bin/brightcove/image\.jpeg\?id=(\d+)&key=', url).groups(1)[0] for url in modal_video_ids]
        video_ids.extend(modal_video_ids)

        playlist_ids = list(set(content.css('div.brightcove-container>video').xpath('./@data-playlist-id').getall()))

        if video_ids or playlist_ids:
            yield {str(host): {'video': video_ids, 'playlist': playlist_ids}}

        return

    def parse_related_products(self, response):
        """
        Identify PDPs that have a Related Products panel
        """
        host = self.check_host(response)
        if not host:
            return None

        try:
            # Run the Javascript with Selenium
            self.driver.get(str(host))
            WebDriverWait(self.driver, timeout=300, poll_frequency=1).until(presence_of_element_located((By.CSS_SELECTOR, 'div.pdp-pod-card')))
        except TimeoutException:
            pass
        body = self.driver.page_source
        body = self.re_spaces.sub(' ', body)
        soup = BeautifulSoup(body, features='lxml')

        content = Selector(text=str(soup))
        if content.xpath('.//div/picture//img[@title="related-product"]'):
            yield {str(host): True}
        return

    def parse_heading_type(self, response):
        """
        Identify H1 type for Touch pages: pageheading or pageheadinghero
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response, exclude_nav=True)

        # Content -headers -left nav
        content = sel.css(self.css_main) \
            .xpath('.//*[not(ancestor-or-self::div[@id="globalHeaderInclusion"])]') \
            .xpath('.//*[not(ancestor-or-self::div[@class="container-leftnav"])]') \
            .xpath('.//*[not(ancestor-or-self::div[@class="cmp-anchorlist"])]')

        map = dict(hero='cmp-p-pageheadinghero', nohero='cmp-p-pageheading')
        matches = {}
        for heading_type, _class in map.items():
            if len(content.css(f'div.{_class}').xpath('.//h1[@class="cmp-pageheading__text"]')) > 0:
                matches[heading_type] = True

        if matches:
            yield {str(host): matches}

        return

    def parse_inbound_links(self, response):
        """
        Get list of pages linking to specified target page
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response, exclude_nav=True)

        # Content -headers -left nav
        content = sel.css(self.css_main) \
            .xpath('.//*[not(ancestor-or-self::div[@id="globalHeaderInclusion"])]') \
            .xpath('.//*[not(ancestor-or-self::div[@class="container-leftnav"])]') \
            # .xpath('.//*[not(ancestor-or-self::div[@class="cmp-anchorlist"])]')

        self.exact = bool(strtobool(str(getattr(self, 'exact', False))))

        if hasattr(self, 'target_url'):
            matches = [link for link in self.normalize_links(content.xpath('.//a/@href|.//script/@src'), host, ignore_scheme=True, strip_suffix=False, as_url=True)
                       if (not self.exact and str(self.target_url) in str(link)) or (self.exact and self.target_url.resolve() == link.resolve())]
            for link in matches:
                yield {str(host): str(link)}

        if hasattr(self, 'target_text'):
            if any([self.target_text.lower() in text.lower() for text in content.xpath('.//text()').getall()]):
                yield {str(host): True}

        return

    def parse_cq_to_pdp(self, response):
        """
        Extract SKUS (links to PDPs)
        """
        # Extract CQ pages with links to PDP
        host = self.check_host(response)
        if not host:
            return None

        check_broken_links = False
        collect_parents = False  # Get parents of CQ pages as well?

        body, soup, sel = self.parse_page(response, exclude_nav=True)

        bespoke_products = []
        if self.is_tfrefresh(sel):
            try:
                # For TF Refresh pages, run the Javascript with Selenium
                t1 = dt.now()
                self.driver.get(str(host))
                WebDriverWait(self.driver, timeout=300, poll_frequency=1).until(presence_of_element_located((By.CSS_SELECTOR, 'div.no-results')))
                t2 = dt.now()
                self.logger.debug(f'{host} loaded in {(t2 - t1).seconds} seconds')
            except TimeoutException:
                self.logger.warning(f'No products found for {host}')
            body = self.driver.page_source
            body = self.re_spaces.sub(' ', body)
            sel = Selector(text=str(self.exclude_nav(body)))

            # Use Selenium to get only visible links
            bespoke_products = [link.get_attribute('href') for link in
                                self.driver.find_elements_by_xpath('.//div[@class="tf-c-table-row"]//a')
                                if link.is_displayed()]
            bespoke_products = [link for link in bespoke_products if link]

            if bespoke_products:
                self.logger.debug(f'{len(bespoke_products)} products found for TFR {host}')

        content = sel.css(self.css_main).xpath('.//*[not(ancestor-or-self::div[@id="globalHeaderInclusion"])]')
        left_nav = sel.css(self.css_left_nav)

        target_types = {'pdp': self.pdplink_xpath,
                        'pcp': self.pcplink_xpath,
                        'fc name': self.collect_cg_products_by_name,
                        'fc id': self.collect_cg_products_by_id,
                        }
        style_types = {
            'title': [content.css('h1'), content.xpath('.//div[@class="h1"]')],
            'inline': [content.xpath('.//*[not(ancestor-or-self::table)]').xpath('.//p|.//li')],
            'table': [content.css('table.rte-table'), content.xpath('.//table//tr/td')],
            'plc': [self.plc_products(content, host)],
            'heading': [content.css('h2,h3,h4'), content.xpath('.//div[@class="h2" or @class="h3" or @class="h4"]')],
            'left nav': [left_nav.css('li.active')],  # Only look at the active subtree (the one with expanded child nodes) - ignores siblings
            'deprecated': [content.xpath(self.deprecated_xpath), left_nav.xpath(self.deprecated_xpath)]}

        key_url = 'url'
        key_style_type = 'style type'
        key_target_type = 'target type'

        links_df = pd.DataFrame(columns=[key_url, key_style_type, key_target_type])
        for style_type, st_collectors in style_types.items():
            for st_collector in st_collectors:
                for target_type, tt_collector in target_types.items():
                    if callable(tt_collector):
                        links = tt_collector(st_collector, host)
                    else:
                        links = self.normalize_links(st_collector.xpath(tt_collector).xpath('@href'), host)
                    links_df = links_df.append([{key_url: link, key_style_type: style_type, key_target_type: target_type} for link in links], ignore_index=True)

        # Bespoke
        links_df = links_df.append([{key_url: link, key_style_type: 'bespoke', key_target_type: 'pdp'} for link in bespoke_products], ignore_index=True)

        # Primary Hijacks
        links_df = links_df.append([{key_url: link, key_style_type: 'implicit', key_target_type: 'primary hijack'} for link in self.detect_primary_hijacks(content, host)], ignore_index=True)

        # Secondary Hijacks
        # links_df = links_df.append([{key_url: link, key_style_type: 'implicit', key_target_type: 'secondary hijack'} for link in self.detect_secondary_hijacks(content, host)], ignore_index=True)

        # Product Selection Guide Featured Collections
        links_df = links_df.append([{key_url: fc_id, key_style_type: 'fc-guide', key_target_type: 'pdp'} for fc_id in set(content.xpath('.//div[@class="cmp-productselectionguide"]/@id').getall())],
                                   ignore_index=True)
        if not links_df[links_df['style type'] == 'fc-guide'].empty:
            # If fc-guide is found, do not collect any direct PDPs (delete them)
            links_df = links_df[~(links_df['style type'] != 'fc-guide') & (links_df['target type'] == 'pdp')]

        parent = sel.css('ul.breadcrumb').xpath('./li[last()-1]/a/@href').get()
        if collect_parents and not parent:
            self.logger.warning(f'No parent found for {response.url}')

        throttle = self.get_throttle()
        if check_broken_links:
            links_df['broken'] = False
            self.logger.info('Testing for broken links')
            for link in links_df[links_df[key_target_type] == 'pdp'][key_url].unique():
                with throttle:
                    resp = requests.head(link)
                    if resp.status_code >= 400 or URL(resp.url).stem == 'error404':
                        self.logger.warning(f'Bad request [{resp.status_code}]: {link}')
                        links_df.loc[links_df[key_url] == link, 'broken'] = True

        out = dict()
        links_df.drop_duplicates(inplace=True)
        if not links_df.empty:
            out = {'links': links_df.to_dict(orient='records')}

        redirect_urls = response.request.meta.get('redirect_urls')
        if redirect_urls:
            out['redirected from'] = redirect_urls

        if out:
            yield {str(host): out}

        if collect_parents and parent and out:
            yield {parent: {k + '_parent': v for k, v in out.items()}}
        return

    def parse_faq(self, response):
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        title = sel.css('title::text').get()
        if any(t in title for t in ['FAQ', 'Frequently Asked Question']):
            qs = {}
            content = sel.css('div.main')
            questions = content.xpath('.//td[.//b]')
            for question in questions:
                question_parts = question.xpath('.//p')
                if len(question_parts) == 2:
                    q, a = question_parts
                    q = q.xpath('.//b/text()').get()
                    a = a.xpath('.//text()').get()
                    qs[q] = a
            yield {host: qs}

    def parse_links(self, response):
        """
        Collect information about links
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        referer = response.request.headers.get('Referer', None)
        if referer:
            referer = referer.decode('utf-8')

        all_links = []

        # Collection links to related applications
        pdp_badge = sel.css('div.pdp-badge')
        if pdp_badge:
            app_links = [str((host / URL(link)).resolve()) for link in pdp_badge.xpath('.//a/@href').getall()]  # TODO: normalize_links instead of resolve()
            invalid = False
        else:
            app_links = []
            invalid = True
        all_links.extend(app_links)

        # Collect links in the product description
        description_links = [str((host / URL(link)).resolve()) for link in sel.css('div[id=pdp-description-wrapper]').xpath('.//a/@href').getall()]  # TODO: normalize_links instead of resolve()
        all_links.extend(description_links)

        # Collect broken links
        broken_links = []
        for link in all_links:
            resp = requests.get(link)
            if resp.status_code != requests.codes.ok or URL(resp.url).stem == 'error404':
                self.logger.warning(f'Bad request: {link}')
                broken_links.append(link)

        d = {'referer': referer, 'applications': app_links, 'description': description_links}
        if broken_links:
            d['broken'] = broken_links
        if invalid:
            d['invalid'] = True
        yield {str(host): d}

    def parse_tools3p(self, response):
        """
        Identify Third Party Tools
        - iFrames
        - External links (one-hop only)
        - <app-categories>?
          https://www.thermofisher.com/us/en/home/life-science/lab-plasticware-supplies/nalgene-bottle-carboy-vial-selection-guide.html
        """
        link_keywords_include = ['3d', 'tour', 'guide', 'virtual', 'interactive', 'tool', 'demo', 'explore', 'animation', 'brochure']
        link_keyword_exclude = ['request', 'download', 'order']
        link_url_exclude = [re.compile('/reference-components/', re.IGNORECASE), re.compile(r'resource\.thermofisher.com/.*?/WF')]
        domains_exclude = ['facebook', 'youtube', 'youtu.be', 'google', 'youku.com', 'brightcove']

        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        content = sel.css(self.css_main)
        tools3p = []

        yield from self.follow_hidden_links(content, host)

        # iFrames
        iframe_urls = self.normalize_links(content.css('iframe::attr(src)'), host)
        throttle = self.get_throttle()
        for iframe_url in iframe_urls:
            with throttle:
                self.logger.info(f'Crawling iframe {iframe_url}')
                resp = requests.get(iframe_url)
                if resp.status_code == requests.codes.ok:
                    soup = BeautifulSoup(resp.text, features='lxml')
                    title = Selector(text=str(soup)).css('head>title').xpath('normalize-space(./text())').get()
                    if not any(domain in iframe_url.lower() for domain in domains_exclude) \
                            and not any(exclude_re.search(iframe_url) for exclude_re in link_url_exclude):
                        tools3p.append({'name': title, 'iframe': iframe_url})

        # Links by keyword
        for link in content.xpath('.//a[not(contains(@href,".pdf"))]'):
            text = {'title': link.xpath('.//@title').get(), 'alt': link.xpath('.//@alt').get(), 'text': link.xpath('normalize-space(.//text())').get()}
            desc = {attribute: value for attribute, value in text.items()
                    if value is not None
                    and any(keyword_include in value.lower() for keyword_include in link_keywords_include)
                    and not any(keyword_exclude in value.lower() for keyword_exclude in link_keyword_exclude)
                    and not any(exclude_re.search(value) for exclude_re in link_url_exclude)}
            if len(desc.values()) > 0:
                tools3p.append({'name': desc.get(list(desc.keys())[0], ''), 'link': self.normalize_links(link.xpath('@href'), host)[0]})

        if tools3p:
            yield {str(host): tools3p}

    def parse_subtitles(self, response):
        """
        Extract subtitles
        - All H1
        - First H2 or H3 that occurs before first paragraph of text
        - First one or two <p> blocks
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        values = dict()

        h1 = sel.css(self.css_main).xpath('normalize-space(.//h1/text())').getall()[:2]  # Get up to 2 H1s

        p = sel.css(self.css_main).xpath('.//p | .//div[@class="h3"]')[:2]  # Paragraph is p or div class="h3"
        p1 = None
        p2 = None
        if p:
            # H2 or H3 preceding first paragraph of text
            subtitle = ''.join(p[0].xpath('preceding::h2 | preceding::h3').xpath('.//text()').getall())
            subtitle = self.trim(subtitle)
            p1 = p[0].xpath('normalize-space(text())').get()
            if len(p) >= 2:
                p2 = (p[1].xpath('normalize-space(text())').get() or '')
        else:
            # Just get first H2 or H3 in content
            subtitles = sel.css(self.css_main).xpath('.//h2 | .//h3').xpath('.//text()')
            if subtitles:
                subtitle = [self.trim(sub.get()) for sub in subtitles if self.trim(sub.get()) != ''][0]
            else:
                subtitle = ''

        if h1:
            values['H1-1'] = h1[0]
            if len(h1) > 1:
                values['H1-2'] = h1[1]  # Second H1
        if subtitle:
            values['Subtitle'] = subtitle
        if p1:
            values['p1'] = p1
        if p2:
            values['p2'] = p2

        yield {str(host): values}

    def parse_metatitles(self, response):
        """
        Get all the meta titles
        """
        host = self.check_host(response)
        if not host:
            return None

        body = response.body.decode('utf-8')
        body = self.re_spaces.sub(' ', body)

        soup = BeautifulSoup(body, features='lxml')
        sel = Selector(text=str(soup))

        title = sel.css('title').xpath('./text()').get()

        yield {str(host): title}

    def parse_text(self, response):
        """
        Get the full text of matching pages
        """
        host = self.check_host(response)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        title = sel.css('head>title').xpath('normalize-space(text())').get()
        title = re.sub(r'(\s+\|\s*Thermo.*$)|(\s+-\s*US$)', '', title)

        # Get text of all elements except <style> and <script> tags, and exclude anything under div.sidebar or div.breadcrumb (recursively)
        text = sel.css(self.css_main).xpath(
            'set:difference(.//*[(not(name()="style") and not(name()="script"))]/text(),.//div[(contains(@class,"sidebar")) or (contains(@class,"breadcrumb"))]//text())').getall()

        # Keep only values containing alphabetic characters
        # Remove escape chars and trim spaces
        text = [re.sub('[\n\r]', ' ', t).strip() for t in text if re.search('[A-z]', t)]

        yield {str(host): {'title': title, 'text': ' '.join(text)}}  # Concatenate all values, separated by space}

    @staticmethod
    def aem_titles(sel):
        sel_titles = [sel.css('head>title').xpath('normalize-space(text())'), sel.xpath('.//meta[@property="og:title"]/@content'), sel.xpath('.//meta[@name="DC.title"]/@content')]
        titles = [re.sub(r'(\s+\|\s*Thermo.*$)|(\s+-\s*US$)', '', st.get()) for st in sel_titles if st]  # Truncate TF suffix
        return titles

    @staticmethod
    def aem_meta_description(sel):
        return sel.xpath('.//meta[@name="description"]/@content').get('')

    def parse_properties(self, response):
        """
        Collect web properties to load in Nucleus
        """
        host = self.check_host(response, strip_query=False)
        if not host:
            return None
        _, _, sel = self.parse_page(response, exclude_nav=True)
        content = sel.css(self.css_main)

        fields = {'web_path': str(host)}
        fields['in_web_us'] = True

        # Title
        titles = self.aem_titles(sel)
        if titles:
            main_title = titles.pop(0)
            fields['web_title'] = main_title
            alternate_titles = ','.join(list(set([t for t in titles if t != main_title])))
            if alternate_titles:
                fields['web_alternate_titles'] = alternate_titles

                # Breadcrumb
        breadcrumb = content.css('div.breadcrumb').xpath('.//li[last()]/span').xpath('normalize-space(.//text())').get()
        fields['web_breadcrumb'] = breadcrumb

        # Subtitle
        subtitle = [s for s in content.xpath('.//h2').xpath('normalize-space(.//text())').getall() if s.strip() != '']
        subtitle = subtitle[0] if subtitle else None
        fields['web_subtitle'] = subtitle

        # H1
        h1 = [s for s in content.xpath('.//h1').xpath('normalize-space(.//text())').getall() if s.strip() != '']
        h1 = h1[0] if h1 else None
        fields['web_h1'] = h1

        # Intro Paragraph
        # H3 or first <p>, whichever comes first
        pars = [s for s in content.xpath('.//div[@class="h3"] | .//p').xpath('normalize-space(.//text())').getall() if s.strip() != '']
        intro_par = pars[0] if pars else None
        fields['web_intro_par'] = intro_par

        # Meta Description
        fields['web_metadescription'] = self.aem_meta_description(sel)

        # Meta Keywords
        metakeywords = sel.xpath('.//meta[@name="keywords"]/@content').get()
        fields['web_metakeywords'] = metakeywords

        # Robots
        robots = sel.xpath('.//meta[@name="robots"]/@content').get()
        fields['web_robots'] = robots

        # Analytics Page Type
        pagetype = sel.xpath('//script[contains(text(),"digitalData.setPageType")]').get()
        try:
            fields['web_analytics_pagetype'] = re.search(r'digitalData.setPageType\("(.*)"\)', pagetype).groups()[0]
        except Exception:
            pass

        # CQ Template
        cqtemplate = sel.xpath('//script[contains(text(),"digitalData.setPageAttribute")]').get()
        try:
            fields['web_cqtemplate'] = re.search(r'digitalData.setPageAttribute\(\'cqtemplate\', "(.*)"', cqtemplate).groups()[0]
        except Exception:
            pass

        # Strip all strings
        for k, v in fields.items():
            if isinstance(v, str):
                fields[k] = v.strip()
        fields = {k: v for k, v in fields.items() if v}

        yield fields

    @staticmethod
    def find_trademarks(tms, host, style_types):
        trademarks = dict()
        for tm, tm_re in tms.items():
            case_sensitive = not tm_re.flags & re.IGNORECASE
            stl_types = set()
            for style_type, st_collectors in style_types.items():
                # Rough but fast search
                if (case_sensitive and any([tm in c for c in st_collectors])) or (not case_sensitive and any([tm.lower() in c for c in map(str.lower, st_collectors)])):
                    # Accurate but slow RE search
                    if any([tm_re.search(m) for m in st_collectors]):
                        stl_types.add(style_type)
            if stl_types:
                trademarks[tm] = {'style type': list(stl_types)}
        if trademarks:
            return {str(host): trademarks}

    def parse_trademarks(self, response):
        """
        Collect Trademarks
        """
        host = self.check_host(response, strip_query=False)
        if not host:
            return None
        _, _, sel = self.parse_page(response, exclude_nav=True, exclude_pdps=True)  # Exclude Trademarks in links so they won't double count
        content = sel.css(self.css_main)

        style_types = {'inline': content.xpath('.//*[not(ancestor-or-self::table)]').xpath('.//p|.//li').xpath('.//text()').extract(),
                       'table': content.css('table.rte-table').xpath('.//text()').extract() + content.xpath('.//table//tr/td').xpath('.//text()').extract(),
                       'plc': self.plc_products(content, host).xpath('.//text()').extract(),
                       'heading': content.css('h2,h3,h4').xpath('.//text()').extract() + content.xpath('.//div[@class="h2" or @class="h3" or @class="h4"]').xpath('.//text()').extract(),
                       'title': self.aem_titles(sel) + content.css('h1').xpath('.//text()').extract() + content.xpath('.//div[@class="h1"]').xpath('.//text()').extract(),  # Title + H1
                       'description': [self.aem_meta_description(sel)]
                       }

        # Dedupe and clean up
        style_types = {k: map(str.strip, v) for k, v in style_types.items() if v}
        style_types = {k: {t for t in v if t} for k, v in style_types.items()}
        style_types = {k: v for k, v in style_types.items() if v}

        # yield self.find_trademarks(self.tms, host, style_types)  # For non-mp testing
        yield from self.mp.run(self.find_trademarks, self.tm_chunks, host=host, style_types=style_types, disable=False)

    def parse_table_titles(self, response: HtmlResponse) -> Any:
        """
        Collect Table Titles
        """
        # Title prefixes that may indicate CVC (product alternatives)
        cvc_title_prefixes = ['choose', 'select', 'selection', 'which']

        host = self.check_host(response, strip_query=False)
        if not host:
            return None
        body, soup, sel = self.parse_page(response)

        # Extract table names for tables with links to at least one product
        tables = dict()
        for table in sel.css('table'):
            product_links = table.xpath(".//a[contains(@href, '/order/catalog/product')]")
            if product_links:
                # Find the first h2 tag before the table
                first_h2_before_table = table.xpath("preceding::h2[1]")
                if first_h2_before_table:
                    table_name = first_h2_before_table.xpath("normalize-space()").extract_first()
                    tables[table_name] = None
                    if any(table_name.lower().startswith(p) for p in cvc_title_prefixes):
                        tables[table_name] = set(product_links.xpath('@href').getall())

        # Return the table names along with the URL of the page
        if tables:
            yield {
                str(host): tables
            }
