.PHONY: population plots finite-sample tests docs-test

population:
	python experiments/identification/run_population_only.py
	python experiments/identification/plot_results.py

plots:
	python experiments/identification/plot_results.py

finite-sample:
	python experiments/identification/run_full_simulation.py

nfxp:
	python -u -c "import sys; sys.path.append('experiments'); import identification.run_nfxp_only as r; r.main()"

tests:
	pytest -q tests/test_appendix_metrics.py

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
