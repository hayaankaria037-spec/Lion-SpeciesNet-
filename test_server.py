"""HTTP adapter tests use explicitly synthetic inference; not model accuracy tests."""
import asyncio
import io
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import server

class AdapterTests(unittest.TestCase):
    def setUp(self):
        server.app.state.token = 'fixture-token-for-private-service'
        server.app.state.lock = asyncio.Lock()
        self.client = TestClient(server.app)
        self.headers = {'Authorization': 'Bearer fixture-token-for-private-service'}

    def test_auth_and_input_rejection(self):
        self.assertEqual(self.client.post('/analyze', content=b'x').status_code, 401)
        self.assertEqual(self.client.post('/analyze', content=b'not an image', headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post('/analyze', content=b'', headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post('/analyze', content=b'x'*(server.MAX_BYTES+1), headers=self.headers).status_code, 413)

    def test_public_readiness_has_no_model_or_secret_details(self):
        server.app.state.model = None
        self.assertEqual(self.client.get('/ready').status_code, 503)
        server.app.state.model = object()
        result = self.client.get('/ready')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {'ready': True})
        self.assertEqual(self.client.get('/health').status_code, 401)

    def test_original_scores_and_failures(self):
        expected = {'prediction_score': .912345, 'prediction': 'fixture lion', 'detections': []}
        with patch.object(server, 'infer', return_value=expected):
            r = self.client.post('/analyze', content=b'fixture', headers=self.headers)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['prediction_score'], .912345)
        with patch.object(server, 'infer', side_effect=RuntimeError('private internal failure')):
            r = self.client.post('/analyze', content=b'fixture', headers=self.headers)
            self.assertEqual(r.status_code, 503)
            self.assertNotIn('private internal', r.text)

if __name__ == '__main__':
    unittest.main()
