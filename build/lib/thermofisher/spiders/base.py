import datetime as dt
import inspect
import logging
import multiprocessing
import os
import platform
import re
import lxml.html
from concurrent.futures import as_completed, ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from distutils.util import strtobool
from itertools import product, islice
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import pyperclip
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from ratelimiter import RateLimiter
from scrapy import Selector
from scrapy.http import Request
from scrapy.selector import SelectorList
from scrapy.spiders import CrawlSpider
from scrapy.utils.project import get_project_settings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.remote_connection import LOGGER as selenium_logger
from tqdm import tqdm
from urlpath import URL

logging.getLogger("scrapy.spidermiddlewares.depth").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Only show warnings
logging.getLogger("urllib3").propagate = False  # Disable all child loggers of urllib3, e.g. urllib3.connectionpool
selenium_logger.setLevel(logging.WARNING)


def isdebugging():
    return any(frame.filename.endswith('pydevd.py') for frame in inspect.stack())


@dataclass
class PerfTest:
    workers_from: int = 10
    workers_to: int = 200
    workers_step: int = 10
    chunks_from: int = 1
    chunks_to: int = 500
    chunks_step: int = 10
    max: int = 500  # Number of rows to test. Just enough for benchmarking

    def __iter__(self):
        # Support unpacking
        return iter((self.workers_from, self.workers_to, self.workers_step, self.chunks_from, self.chunks_to, self.chunks_step, self.max))


class Multiprocessing:
    def __init__(self):
        """
        Initialize a multiprocessing pool that excludes the CPU running sshd and busy CPUs
        """
        cpus = list(range(psutil.cpu_count(logical=True)))
        if platform.system() != 'Windows':
            # Exclude sshd CPU
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'sshd':
                    sshd_cpu_num = psutil.Process(proc.info['pid']).cpu_num()
                    print(f'Excluding CPU #{sshd_cpu_num} running sshd from multiprocess pool')
                    cpus.remove(sshd_cpu_num)
                    break

            # Exclude busy CPUs
            threshold = 20  # Anything > 20% is too busy
            cpu_loads = psutil.cpu_percent(interval=1, percpu=True)
            high_load_cpus = [i for i, load in enumerate(cpu_loads) if load > threshold]
            if high_load_cpus:
                print(f'Excluding CPUs {high_load_cpus} with load average >{threshold}% from multiprocess pool')
                cpus = [cpu for cpu in cpus if cpu not in high_load_cpus]

            os.sched_setaffinity(0, cpus)  # The current process (PID 0) can only run on these cpus

        self.n_cpus = len(cpus)
        self.pool = ProcessPoolExecutor(max_workers=self.n_cpus)
        return

    def run(self, func, chunks: List, disable=False, *args, **kwargs):
        """
        Function multiprocessing
        :param func: Function to multiprocess. The function's first argument must be a data chunk
        :param chunks: List of partitioned data chunks to farm out
        :param disable: Disable MP and just run serially. Good for testing.
        :param args: Function args
        :param kwargs: Function kwargs
        :return: List of results
        """

        if not disable:
            for future in as_completed(self.pool.submit(func, chunk, *args, **kwargs) for chunk in chunks):
                result = future.result()
                if result:
                    yield result
        else:
            results = []
            for chunk in chunks:
                results.append(func(chunk, *args, **kwargs))


class BaseSpider(CrawlSpider):
    name = None
    allowed_domains = []
    logger = logging.getLogger()
    settings = get_project_settings().attributes
    throttle_delay = 1  # seconds
    custom_settings = {'ROBOTSTXT_OBEY': False}

    # Normalize punctuation
    re_spaces = re.compile(r'(\xa0|#nbsp;|#160;)')
    re_single_quote = re.compile(r"(`|\u2018|\u2019|\u201b|\u2032)")
    re_double_quote = re.compile(r'(\u00ab|\u00bb|\u201c|\u201d|\u201f\u2033)')
    re_dot = re.compile(r'(\u00b7)')

    # Ajax code
    re_ajax = re.compile(r"\.load\('([^']+)'")

    # Old style URLs
    re_adirect_sku = re.compile(r'/\w{2}/\w{2}/adirect/lt\?cmd=catProductDetail&.*?productID=(.+?)\b.*', re.IGNORECASE)
    re_adirect_category = re.compile(r'/order/catalog/\w{2}/\w{2}/adirect/lt\?cmd=.*?DisplayCategory.*?&catKey=(.+?)\b.*', re.IGNORECASE)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.logger.info(f'PID: {os.getpid()}')
        self.mp = Multiprocessing()

        if isdebugging():  # Help to troubleshoot only specified start URLs
            self.custom_settings['DEPTH_LIMIT'] = 1
        self.pandas_options()
        self.thread_executor = ThreadPoolExecutor(max_workers=100)
        self.base_dir = Path('.') / self.name

        if hasattr(self, 'cached') and strtobool(self.cached):
            # Disable all throttling if scraping local cache
            print('Crawling with Max Performance using cache')
            n_threads = (multiprocessing.cpu_count() * 2) - 1
            self.custom_settings['DOWNLOAD_DELAY'] = 0
            self.custom_settings['AUTOTHROTTLE_ENABLED'] = False
            self.custom_settings['CONCURRENT_ITEMS'] = 100
            self.custom_settings['CONCURRENT_REQUESTS'] = n_threads
            self.custom_settings['CONCURRENT_REQUESTS_PER_DOMAIN'] = n_threads
            self.custom_settings['CONCURRENT_REQUESTS_PER_IP'] = 0

        # Selenium Firefox Web Driver
        options = webdriver.FirefoxOptions()
        options.headless = True
        geckodriver = 'geckodriver' + ('.exe' if os.name == 'nt' else '')
        self.driver = webdriver.Firefox(service=Service(geckodriver), options=options)

        return

    @staticmethod
    def isdebugging():
        return any(frame.filename.endswith('pydevd.py') for frame in inspect.stack())

    def extracted_links(self, response):
        """
        :return: Links extracted from current response
        """
        return self.rules[0].link_extractor.extract_links(response)

    def get_throttle(self):
        # Throttler
        return RateLimiter(max_calls=1, period=self.throttle_delay)

    def write_string_to_file(str, filename):
        with open(filename, 'wt', encoding='utf-8') as f:
            f.write(str)

    @staticmethod
    def pandas_options():
        """
        Set Pandas console parameters for development
        """
        pd.options.mode.chained_assignment = None
        np.set_printoptions(formatter={'float_kind': '{:.4f}'.format}, linewidth=200, edgeitems=5)  # Limit float display to 4 decimals
        pd.options.display.max_rows = 20
        pd.options.display.max_columns = 20
        pd.options.display.max_colwidth = 80
        pd.options.display.width = 450
        pd.set_option('display.float_format', lambda x: f'{x:.4f}')
        return

    @staticmethod
    def toclip(string):
        """
        Copy string to clipboard on Windows
        """
        pyperclip.copy(str(string))

    def parse_start_url(self, response):
        """
        CrawlSpider calls parse_start_url for start URLs instead of our callback
        But we want those parsed as well
        Force start URLs to the defined callback
        """
        yield Request(response.url, getattr(self, self.rules[0].callback))

    def check_host(self, response, strip_query=True, strip_fragment=True) -> URL:
        """
        Normalize host and return as URL object
        :param response: The Scrapy Response object
        :return: Normalized host as URL object if page is valid. Otherwise None.
        """

        # Only accept text/html
        if b'text/html' not in response.headers.get('Content-Type', ''):
            self.logger.warning('Skipping non-HTML content')
            return None

        host = re.sub(r'(\.html?){2,}$', '.html', response.url)  # Replace multiple .htm(l) suffixes with one
        host = URL(host.strip('/'))
        if strip_query:
            host = host.with_query('')
        if strip_fragment:
            host = host.with_fragment('')

        # Check host URL against the Rules.
        # Scrapy does not do this for redirect targets so we must do this manually
        for rule in self.rules:
            # Allow
            if rule.link_extractor.allow_res and not any([allow_re.search(str(host)) for allow_re in rule.link_extractor.allow_res]):
                self.logger.warning(f'{str(host)} not allowed. Skipping')
                return None
            # Deny
            if any([deny_re.search(str(host)) for deny_re in rule.link_extractor.deny_res]):
                self.logger.warning(f'{str(host)} denied. Skipping')
                return None
            # Allow Subdomains
            if rule.link_extractor.allow_domains and not any([host.netloc.endswith(allow_domain) for allow_domain in rule.link_extractor.allow_domains]):
                self.logger.warning(f'{host.netloc} not allowed subdomain. Skipping')
                return None
            # Deny Subdomains
            if rule.link_extractor.deny_domains and any([host.netloc.endswith(deny_domain) for deny_domain in rule.link_extractor.deny_domains]):
                self.logger.warning(f'{host.netloc} denied subdomain. Skipping')
                return None
            # Deny Extensions
            if any(suffix.lower() in rule.link_extractor.deny_extensions for suffix in host.suffixes):
                self.logger.warning(f'{host.suffix} extension denied. Skipping')
                return None

        return host

    def normalize_punctuation(self, txt):
        txt = self.re_spaces.sub(' ', txt)
        txt = self.re_single_quote.sub("'", txt)
        txt = self.re_double_quote.sub('"', txt)
        txt = self.re_dot.sub('.', txt)
        return txt

    def exclude_nav(self, body):
        soup = BeautifulSoup(body, features='lxml')
        # Exclude nav content
        for id in self.ids_exclude:
            for tag in soup.select(f'div[id="{id}"]'):
                tag.decompose()
        return soup

    def exclude_pdps(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Selects elements from a BeautifulSoup object using an XPath expression.
        """
        root = lxml.html.fromstring(str(soup))
        elements = root.xpath(self.pdplink_xpath)
        for element in ResultSet(root, elements):
            for tag in soup.find_all(href=element.get('href')):
                tag.decompose()
        return soup

    def parse_page(self, response, exclude_nav: bool = False, exclude_pdps: bool = False):
        """
        Prepare the page for parsing
        """
        body = response.body.decode('utf-8')
        body = self.normalize_punctuation(body)
        soup = BeautifulSoup(body, features='lxml')

        if exclude_nav:
            soup = self.exclude_nav(body)

        if exclude_pdps:
            soup = self.exclude_pdps(soup)
        sel = Selector(text=str(soup))
        return body, soup, sel

    @staticmethod
    def trim(str):
        # Remove whitespace
        if str is not None:
            return re.sub(r'\s+', ' ', str).strip()
        else:
            return ''

    def normalize_links(
            self,
            selector: Union[Selector, SelectorList, List],
            host,
            ignore_scheme=False,
            strip_query=True,
            strip_fragment=True,
            strip_suffix=False,
            as_url=False,
            https=True
    ) -> list:
        """
        Fix links
        Convert relative to absolute (using host)
        Map obsolete links to new format
        Optionally strip queries and fragments
        Dedupe

        :param selector: Selector or List containing URLs to normalize
        :param host: Host that relative links are relative to. Used to conver to absolute
        :param ignore_scheme: Ignore scheme (http/https)
        :param strip_query: Strip query strings (?param=value)
        :param strip_fragment: Strip page fragments (# anchors)
        :param strip_suffix: Strip suffix (.htm/.html)
        :param as_url: Return links as URL objects, otherwise strings
        :param https: Normalize all links to https
        :return:
        """
        if isinstance(host, str):
            host = URL(host)

        # Normalize product link URLs
        if isinstance(selector, SelectorList) or isinstance(selector, Selector):
            links = set(selector.getall())
        elif isinstance(selector, List):
            links = selector
        else:
            raise TypeError('selector must be Selector, SelectorList or list')

        # Strip leading and trailing spaces
        links = [link.strip() for link in links]

        if not ignore_scheme:
            scheme = 'https' if https else host.scheme
            links = [link for link in links if link and (link.startswith('http') or link.startswith('/'))]  # Filter out invalid links (None, javascript, empty etc)
            links = [scheme + ':' + link if link.startswith('//') else link for link in links]  # //www → https://www

        links = [self.re_adirect_sku.sub(r'/product/\1', link) for link in links]  # Map obsolete adirect/lt to product links
        links = [self.re_adirect_category.sub(r'/search/browse/category/us/en/\1', link) for link in links]  # Map obsolete adirect/lt to category links

        # Convert to URL
        def to_url(link):
            try:
                url = URL(link)
                if url == URL(''):
                    return None
                else:
                    return url
            except Exception:
                return None

        links = [url for url in [to_url(link) for link in links] if url]

        # Convert to fully qualified URL
        links = [link.with_netloc(host.netloc) if link.netloc == '' else link for link in links]  # Add netloc if missing

        if not ignore_scheme:
            links = [link.with_scheme(scheme) for link in links]

        # Strip querystring
        if strip_query:
            links = [link.with_query('') for link in links]

        # Strip fragment
        if strip_fragment:
            links = [link.with_fragment('') for link in links]

        # Strip suffix
        if strip_suffix:
            links = [link.with_suffix('') if link.name else link for link in links]

        # Convert URL back to string
        if not as_url:
            links = [str(link) for link in links]

        return list(set(links))

    @staticmethod
    def now():
        return dt.datetime.now()

    @staticmethod
    def chunks(array, chunksize: int = None, start: int = 0, report_bounds: bool = False, progress_desc=None):
        """
        Break a large array into large chunks. Yield a series of indices to iterate through the array
        :param array: Array to chunk
        :param chunksize: Chunk size
        :param start: Start index
        :param report_bounds: Report the indices for each iteration?
        :param progress_desc: TQDM description
        :yield: Iteration Indices
        """
        # TODO: If progress_desc, integrate TQDM
        i = start
        if hasattr(array, 'shape'):
            size = array.shape[0]
        else:
            size = len(array)
        if chunksize and chunksize < size:
            while i < size:
                j = min(i + chunksize, size)
                if report_bounds:
                    print(f'{i:,} - {j:,}')
                yield i, j
                i = j
        else:
            # If chunksize not specified, return whole set
            yield i, size

    def parallel_run(self, func, data, description=None, workers=200, chunksize=1, silent: bool = False, perf_test: PerfTest = None, **kwargs):
        """
        Run a function over a dataset in thread parallel
        :param func: Function that operates over data, taking a sub-collection as input and returning an equivalent sub-collection as output
        :param data: Collection of items to process
        :param description: Progress Bar description. If None, progress bar is not displayed
        :param workers: Number of Worker Threads
        :param chunksize: Number of chunks to break each thread payload into. Default 1 = each thread will process one item at a time
        :param perf_test: Do a performance test by scanning worker threads to find optimal number
        :param silent: Do not display progress bar
        :return: Series of results, corresponding to the data collection
        """

        def func_with_pbar(func, pbar, chunksize):
            """"
            Inject progress bar updates to function
            """

            def inner(*args, **kwargs):
                pbar.update(chunksize)
                return func(*args, **kwargs)

            return inner

        def run(data):
            pool = []
            with tqdm(total=len(data), smoothing=0, desc=f"{'Performance Test: ' if perf_test else ''}{description}", disable=silent or description is None) as pbar:
                pbar.postfix = f'Workers: {workers:,} Chunksize: {chunksize:,}'
                with ThreadPoolExecutor(max_workers=workers) as thread_executor:
                    for i, j in self.chunks(data, chunksize):
                        pool.append(thread_executor.submit(func_with_pbar(func, pbar, chunksize), data[i:j], **kwargs))
                    for future in as_completed(pool):
                        r = future.result()
                        if r:
                            results.append(r)
            return results

        data = list(data)
        results = []
        if perf_test:
            workers_from, workers_to, workers_step, chunks_from, chunks_to, chunks_step, rows = perf_test
            timing = dict()
            for workers, chunksize in product(range(workers_from, workers_to + 1, workers_step), range(chunks_from, chunks_to + 1, chunks_step)):
                t1 = self.now()
                results = run(data[:rows])
                t2 = self.now()
                timing[workers, chunksize] = (t2 - t1).seconds

            timing = pd.DataFrame.from_dict(timing, orient='index').rename(columns={0: 'Seconds'}).rename_axis('Workers')
            timing.index = pd.MultiIndex.from_tuples(timing.index, names=['Workers', 'Chunksize'])
            opt_workers, opt_chunks = timing['Seconds'].idxmin()

            print(timing)
            print('Optimal Settings:')
            print(f'Workers: {opt_workers:,}')
            print(f'Chunksize: {opt_chunks:,}')
            print(f"Performance Improvement: {np.float64(timing['Seconds'].max()) / timing['Seconds'].min():.2}x")

            fig = px.scatter_3d(timing.reset_index(), 'Workers', 'Chunksize', 'Seconds')
            fig.show()
            return timing
        else:
            results = run(data)
        return results

    def chunk_dict(self, d, chunk_size):
        """Chunk a dictionary into smaller dictionaries with the given chunk size."""
        return [{k: d[k] for k in islice(d, i, j)} for i, j in self.chunks(d, chunk_size)]
