VENV = .venv
VENV_ACTIVATE = PATH=$(shell pwd)/$(VENV)/bin
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

install: requirements.txt
	rm -rf $(VENV)
	python3.8 -m venv $(VENV)
	$(PIP) install -r requirements.txt