import os
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, help='target filename')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TARGET_DIR = os.path.join(BASE_DIR, 'fs')
BASE_URL = 'https://www.fishersci.com'

vaild_directories = [
    'amino-acids-reference-tool', 'brands', 'contactus', 'customer-help-support',
    'education-products', 'error', 'footer', 'healthcare-products', 'home',
    'our-response-to-the-covid-19-outbreak', 'periodic-table', 'products', 'programs',
    'science-social-hub', 'scientific-products'
]


def create_start_uri():
    args = parser.parse_args()
    filename = args.filename
    filepath = os.path.join(TARGET_DIR, filename)
    df = pd.read_excel(filepath)
    col_A = df.get('jcr:path')
    col_L = df.get('contentID')

    feature = []

    for c in list(col_A):
        if isinstance(c, str) and c.split('/')[4] in vaild_directories:
            feature.append(c.split('/')[4])
        else:
            feature.append(None)

    content_data = {
        'feature': feature,
        'content_id': list(col_L)
    }
    content_df = pd.DataFrame(content_data)
    is_unique = content_df.groupby('feature')['content_id'].nunique() == content_df.groupby('feature')['content_id'].count()
    unique_dict = is_unique.to_dict()
    if all(value is True for value in unique_dict.values()):
        print('The contentIds are not unique for the green directories')
    else:
        non_unique_vals = content_df[content_df.duplicated('content_id')]['content_id'].tolist()
        false_keys = [k for k, v in unique_dict.items() if v is False]
        print(f'The contentId is not unique: {false_keys}')
        print(f'non-unique values: {list(set(non_unique_vals))}')
    start_uris = [BASE_URL + uri + '.html' for uri in list(col_A) if isinstance(uri, str)]

    with open(os.path.join(TARGET_DIR, 'start_uris.txt'), 'w') as f:
        for uri in start_uris:
            f.write(f'{uri}\n')


create_start_uri()
