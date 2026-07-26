# Joe's sealed environment: a fresh PyPI install, nothing from this repo.
# He sees only what a stranger sees: the published econirl wheel + his tools.
FROM python:3.11-slim

# A real applied scientist's kit: the package, plus notebooks and the usual stack.
RUN pip install --no-cache-dir \
    econirl==0.0.10 \
    jupyter pandas numpy matplotlib

WORKDIR /work
# Inputs (panel.csv, features.npy, problem.json, paper.pdf) are copied in at run
# time by run.sh. truth.json is NEVER copied in.
CMD ["sleep", "infinity"]
