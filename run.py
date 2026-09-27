"""Bind to the hosting provider's assigned port without exposing credentials."""
import os
import uvicorn

if __name__ == '__main__':
    port = int(os.getenv('PORT', '8080'))
    if not 1 <= port <= 65535:
        raise ValueError('PORT must be between 1 and 65535')
    uvicorn.run('server:app', host='0.0.0.0', port=port, workers=1,
                limit_concurrency=8, access_log=False)
