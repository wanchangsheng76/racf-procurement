# RACF: Reputation-Aware Coalition Formation
Code and data for "Endogenous Reputation and Coalition Formation in Decentralized Procurement"

## Data
The BOAMP public procurement dataset is publicly available at DOI: 10.5281/zenodo.11001277 (Deschamps et al. 2025).

## Code Structure
- model.py: Core model (supplier, order, coalition classes)
- racf_stages.py: RACF three-stage mechanism
- experiments.py: Main computational experiments
- experiments_multicycle.py: Multi-cycle feedback loop experiments
- generate_figures.py: Figure generation
- config.py: Parameter configuration
- benchmarks.py: Benchmark implementations
- analysis.py: Statistical analysis
- requirements.txt: Python dependencies

## Reproducibility
pip install -r requirements.txt
python experiments.py
python experiments_multicycle.py
python generate_figures.py
