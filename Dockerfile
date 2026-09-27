FROM python:3.11-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY server.py run.py ./
RUN useradd --create-home inference
USER inference
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SPECIESNET_CPU_THREADS=2
# Cache official weights in the container image and verify they load.
# An incomplete download or invalid checkpoint fails the build rather than faking readiness.
RUN python -c "from server import load_model; load_model()"
EXPOSE 8080
CMD ["python", "run.py"]
