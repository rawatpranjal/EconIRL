.PHONY: tests docs docs-test rust-table-ix ccp-table-ix mce-gridworld distclean build

tests:
	pytest -q

docs:
	python -m sphinx -b html docs docs/_build/html

docs-test:
	python -c "\
	from econirl import NFXP, CCP; \
	from econirl.datasets import load_rust_bus, rust_bus_reward_spec; \
	df = load_rust_bus(); \
	utility = rust_bus_reward_spec(90); \
	nfxp = NFXP(n_states=90, discount=0.9999, utility=utility).fit(df, state='mileage_bin', action='replaced', id='bus_id'); \
	ccp = CCP(n_states=90, discount=0.9999, utility=utility, num_policy_iterations=5).fit(df, state='mileage_bin', action='replaced', id='bus_id'); \
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

ccp-table-ix:
	mkdir -p downloads validation/results
	test -f downloads/nfxp.zip || curl -L -o downloads/nfxp.zip https://editorialexpress.com/jrust/nfxp.zip
	test -f downloads/nfxp_unzip/nfxp/dat/a530875.asc || unzip -q downloads/nfxp.zip -d downloads/nfxp_unzip
	PYTHONPATH=src uv run python validation/estimators/ccp/rust_table_ix.py --raw-path downloads/nfxp_unzip/nfxp/dat/a530875.asc --output validation/results/ccp_rust_table_ix.json
	PYTHONPATH=src uv run python validation/estimators/ccp/rust_table_ix.py --output validation/results/ccp_rust_table_ix.json --verify

mce-gridworld:
	PYTHONPATH=src uv run python examples/ziebart-mce-irl/run_gridworld.py --grid-size 5 --n-traj 500 --n-periods 30

distclean:
	rm -rf dist build *.egg-info

build:
	uv build --out-dir dist
	uvx twine check dist/*.whl dist/*.tar.gz
