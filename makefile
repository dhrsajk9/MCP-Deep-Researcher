.PHONY: install setup test clean run-server search ask stats

# Install dependencies
install:
	pip install -e .
	pip install -r requirements.txt

# Setup project
setup: install
	if not exist "data\papers" mkdir data\papers
	if not exist "data\vectors" mkdir data\vectors

# Clean data
clean:
	if exist "data\vectors" rmdir /s /q data\vectors
	if exist "data\papers" rmdir /s /q data\papers
	if not exist "data\papers" mkdir data\papers
	if not exist "data\vectors" mkdir data\vectors

# Run MCP server
run-server:
	python src/server.py

# Search for papers
search:
	python src/cli.py search -q "$(query)" -n 5

# Ask questions
ask:
	python src/cli.py ask -q "$(question)"

# Show stats
stats:
	python src/cli.py stats

# Test installation
test:
	python -c "import sys; sys.path.insert(0, 'src'); from paper_retriever import PaperRetrieverManager; print('✅ All imports work')"