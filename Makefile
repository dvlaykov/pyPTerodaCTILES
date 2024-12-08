.PHONY: test doc format mypy

test:
	python3 -m tox

format:
	python3 -m tox -e fmt

mypy:
	python3 -m tox -e mypy

style_check:
	python3 -m tox -e style_check