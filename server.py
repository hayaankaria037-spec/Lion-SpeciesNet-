"""Private SpeciesNet HTTP adapter. Uses Google's official package without fake inference."""
import asyncio
import hmac
import io
import os
import tempfile
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL = os.getenv('SPECIESNET_MODEL', 'kaggle:google/speciesnet/pyTorch/v4.0.3a/1')
MAX_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000


def checkpoint(stage):
    stats = {}
    for name in ('memory.current', 'memory.max', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            stats[name] = path.read_text().strip()
    print('SpeciesNet startup:', stage, stats, flush=True)


def load_model():
    checkpoint('before torch import')
    import torch
    checkpoint('after torch import')
    torch.set_num_threads(max(1, min(8, int(os.getenv('SPECIESNET_CPU_THREADS', '2')))))
    from speciesnet import SpeciesNet
    checkpoint('after speciesnet import')
    model = SpeciesNet(MODEL, geofence=False, multiprocessing=False)
    checkpoint('model loaded')
    return model


@asynccontextmanager
async def lifespan(app):
    token = os.getenv('SPECIESNET_TOKEN', '')
    if len(token) < 32:
        raise RuntimeError('Set SPECIESNET_TOKEN to a random secret of at least 32 characters.')
    app.state.token = token
    # Fail startup if official weights cannot load; never advertise readiness without a model.
    app.state.model = await asyncio.to_thread(load_model)
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware('http')
async def protect(request: Request, call_next):
    if request.method == 'GET' and request.url.path == '/ready':
        ready = getattr(app.state, 'model', None) is not None
        return JSONResponse({'ready': ready}, status_code=200 if ready else 503, headers={'Cache-Control': 'no-store'})
    token = getattr(app.state, 'token', '')
    if not token or not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/health')
async def health():
    return {'ready': True, 'model': MODEL, 'packageVersion': version('speciesnet')}


def infer(data: bytes):
    # Decode and normalize uploads; never accept user file paths or fetch arbitrary URLs.
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in ('JPEG', 'PNG') or im.width * im.height > 24_000_000:
            raise ValueError('Use a JPG or PNG with no more than 24 megapixels.')
        im.load()
        with tempfile.TemporaryDirectory(prefix='speciesnet-') as directory:
            path = Path(directory) / 'observation.jpg'
            ImageOps.exif_transpose(im).convert('RGB').save(path, quality=95)
            result = app.state.model.predict(filepaths=[str(path)], run_mode='single_thread')
    rows = result.get('predictions', []) if isinstance(result, dict) else []
    if len(rows) != 1 or rows[0].get('failures') or rows[0].get('failure'):
        raise RuntimeError('Incomplete prediction')
    row = rows[0]
    # Preserve model-produced scores; strip temporary filesystem paths and other internals.
    return {'model': MODEL, 'packageVersion': version('speciesnet'), 'prediction': row.get('prediction'),
            'prediction_score': row.get('prediction_score'), 'prediction_source': row.get('prediction_source'),
            'classifications': row.get('classifications'), 'detections': row.get('detections', [])}


@app.post('/analyze')
async def analyze(request: Request):
    if app.state.lock.locked():
        raise HTTPException(429, 'Model is busy. Retry shortly.')
    async with app.state.lock:
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, 'Maximum image size is 8 MB.')
        if not data:
            raise HTTPException(400, 'An image is required.')
        try:
            return await asyncio.to_thread(infer, bytes(data))
        except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError):
            raise HTTPException(400, 'Invalid JPG/PNG or excessive image dimensions.') from None
        except Exception:
            raise HTTPException(503, 'SpeciesNet could not complete inference. No prediction is available.') from None
