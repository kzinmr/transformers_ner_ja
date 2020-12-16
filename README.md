# transformers_ner_ja
Japanese NER with Transformers + PyTorch-Lightning + MLflow Tracking

## GPU Training
- build: `docker build -t trf-ner-ja-train .`
- run: `docker run --rm --gpus all  -v /where/to/workspace/outputs:/app/workspace trf-ner-ja-train`
  - NOTE: set `export GPUS=1` in run_ner.sh

## MLflow Tracking
- build: `docker-compose build`
- run: `docker-compose up`
  - NOTE: check `./workspace/mlruns/0/xxx` is created for each runs
  - NOTE: GPU support in docker-compose will be released in 1.28.0: See. https://github.com/docker/compose/pull/7929
- view: open http://localhost:5000/ in your browser
