.PHONY: install setup test clean run-server

install:
	pip install -e .
	pip install -r requirements.txt

setup: install
	mkdir -p data/papers data/vectors
	python -c "import nltk; nltk.download('punkt')"

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/vectors/*
	rm -rf data/papers/*
	rm -rf __pycache__/
	rm -rf src/__pycache__/

run-server:
	python src/server.py

search:
	python src/cli.py search -q "$(query)" -n 5

ask:
	python src/cli.py ask -q "$(question)"

stats:
	python src/cli.py stats