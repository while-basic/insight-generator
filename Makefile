.PHONY: test run

test:
	pytest

run:
	chat-recycler run --in data/sample_export --out out
