from setuptools import setup

setup(name='Thermo Fisher Crawler',
      version='2.0',
      author='Steven Nicolaou',
      author_email='steven.nicolaou@thermofisher.com',
      install_requires=['scrapy',
                        'selenium==4.10.0',
                        'bs4',
                        'pandas',
                        'requests',
                        'deprecation',
                        'validators'],
      )

# To install dependencies run as
# python setup.py install
# If scrapy install fails, manually download the Twisted whl from here and pip install that first
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#twisted
