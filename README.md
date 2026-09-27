# Lion SpeciesNet Backend

FastAPI adapter using Google's official SpeciesNet package: https://github.com/google/cameratrapai

Deploy root Dockerfile on Railway. Set SPECIESNET_TOKEN to a random secret of at least 32 characters. PORT is provided by the host. Build downloads and loads official weights; failures stop the build.

GET /ready returns readiness. GET /health and POST /analyze require Authorization: Bearer <token>. Analyze accepts raw JPG/PNG bytes, maximum 8 MB and 24 megapixels. Temporary images are removed after inference.

Outputs preserve actual prediction_score, prediction_source, classifications and detections. Scores are uncalibrated species scores, not behaviour confidence. SpeciesNet does not determine behaviour, health or image freshness. A non-lion top prediction does not rule out lions elsewhere in a multi-animal image.

Keep secrets and observations out of Git. Configure the existing app's server-only SPECIESNET_URL to the deployed /analyze endpoint and matching SPECIESNET_TOKEN after real inference testing. Unit tests use explicit fixtures and do not measure model accuracy.
