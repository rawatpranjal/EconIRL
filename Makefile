.PHONY: tests docs docs-test rust-table-ix mce-gridworld distclean build publish-test publish

tests:
	pytest -q

docs:
	python -m sphinx -b html docs docs/_build/html

docs-test:
	python -c "\
	from econirl import NFXP, CCP; \
	from econirl.datasets import load_rust_bus; \
	df = load_rust_bus(); \
	nfxp = NFXP(discount=0.9999).fit(df, state='mileage_bin', action='replaced', id='bus_id'); \
	ccp = CCP(discount=0.9999, num_policy_iterations=5).fit(df, state='mileage_bin', action='replaced', id='bus_id'); \
	print('params:', nfxp.params_); \
	print('se:', nfxp.se_); \
	import numpy as np; \
	proba = nfxp.predict_proba(np.array([0, 30, 60, 89])); \
	print('proba:', proba); \
	print('Quickstart smoke test passed') \
	"

rust-table-ix:
	mkdir -p downloads acceptance/loop/nfxp/table_ix
	test -f downloads/nfxp.zip || curl -L -o downloads/nfxp.zip https://editorialexpress.com/jrust/nfxp.zip
	test -f downloads/nfxp_unzip/nfxp/dat/a530875.asc || unzip -q downloads/nfxp.zip -d downloads/nfxp_unzip
	PYTHONPATH=src uv run python -m econirl.replication.rust1987.table_ix --raw-path downloads/nfxp_unzip/nfxp/dat/a530875.asc --out acceptance/loop/nfxp/table_ix

mce-gridworld:
	PYTHONPATH=src uv run python examples/ziebart-mce-irl/run_gridworld.py --grid-size 5 --n-traj 500 --n-periods 30

distclean:
	rm -rf dist build *.egg-info

build:
	python -m pip install --upgrade build twine >/dev/null
	python -m build
	twine check dist/*

publish-test:
	@echo "Uploading to TestPyPI (set TWINE_USERNAME=__token__ and TWINE_PASSWORD=***token***)"
	twine upload --repository testpypi dist/*

publish:
	@echo "Uploading to PyPI (set TWINE_USERNAME=__token__ and TWINE_PASSWORD=***token***)"
	twine upload --repository pypi dist/*
