.PHONY: install test lint typecheck run-hkex run-krx run-all clean docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check domain application infrastructure interfaces agents tests config.py

typecheck:
	mypy domain config.py

run-hkex:
	python -m interfaces.cli run --exchange HKEX --days 10

run-krx:
	python -m interfaces.cli run --exchange KRX --days 10

run-all:
	python -m interfaces.cli run --all --days 10

clean:
	rm -rf data/reports/*.html .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

docker-build:
	docker build -t hsi-sentiment-agent .

docker-run:
	docker run --rm -v $$(pwd)/data:/app/data --env-file .env hsi-sentiment-agent python -m interfaces.cli run --all
