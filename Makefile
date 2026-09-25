install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest -q

check-data:
	python scripts/check_dataset.py --root data/EchoNet-Dynamic
