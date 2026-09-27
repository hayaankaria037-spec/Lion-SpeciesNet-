"""Run against the real deployed endpoint. No mocks, no fabricated scores.
Usage: SPECIESNET_URL / SPECIESNET_TOKEN in environment, python smoke_test.py lion.jpg
"""
import io
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from PIL import Image

url = os.environ['SPECIESNET_URL']
token = os.environ['SPECIESNET_TOKEN']
if urlparse(url).scheme != 'https':
    raise ValueError('Use the deployed HTTPS endpoint')

class NoRedirect(__import__('urllib.request', fromlist=['HTTPRedirectHandler']).HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

from urllib.request import build_opener
opener = build_opener(NoRedirect())

def check(name, data):
    req = Request(url, data=data, method='POST', headers={'Authorization': 'Bearer '+token, 'Content-Type': 'image/jpeg'})
    with opener.open(req, timeout=90) as response:
        result = json.load(response)
    score = result.get('prediction_score')
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1):
        raise ValueError('Invalid model score')
    print(json.dumps({'case': name, 'model': result.get('model'), 'prediction': result.get('prediction'), 'raw_score': score}))
    return result

with open(sys.argv[1], 'rb') as file:
    lion = check('supplied lion photograph', file.read())
blank = io.BytesIO()
Image.new('RGB', (512, 512), (128, 128, 128)).save(blank, format='JPEG')
check('plain grey image — no animal', blank.getvalue())
print('Inspect both actual predictions before enabling the Site connection. This is a smoke test, not accuracy calibration.')
