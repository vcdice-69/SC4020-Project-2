# SC4020-Project-2

## Setup (Python venv)

Requires Python 3.11+. From the project root:

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Notebooks:** in VS Code, click **Select Kernel** → **Python Environments** → `.venv`.

## Processed data

Running `task1.ipynb` produces `data/processed/davis_clean.csv` and `data/processed/kiba_clean.csv`. Each row represents one observed ligand–target pair after cleaning and deduplication.

| Column | Description |
|---|---|
| `ligand_id` | Ligand (drug) identifier (DAVIS: PubChem CID; KIBA: ChEMBL ID) |
| `target_id` | Target (protein) identifier (DAVIS: kinase name; KIBA: UniProt accession) |
| `smiles` | Canonical SMILES string of the ligand (drug) |
| `sequence` | Amino-acid sequence of the target protein |
| `score` | Binding affinity. Higher values indicate stronger binding (DAVIS: pKd = 9 − log10(Kd [nM]); KIBA: supplied KIBA score) |

For the purposes of this project, models are built to predict `score`, using `smiles` and `sequence` as inputs. The identifiers `ligand_id` and `target_id` may be used as inputs in place of these for the collaborative filtering approach in Task 2.
