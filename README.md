# Sider
Dual representation learning model for drug-side effect frequency prediction.

# Requirements

```python == 3.6.13
pytorch == 1.10.2
torch-geometric == 2.0.3
numpy == 1.19.5
scipy == 1.5.4
scikit-learn == 0.24.2
networkx == 2.5.1
rdkit == 2020.09.1.0
openpyxl == 3.0.9
```

# Run

## Basic installation
```bash
python setup.py install

# python -c "import sider; dataset = sider.dataset(); model = sider.model()"
```

## Reproducing the results of paper
```bash
python main.py --split drug --drug_boost
```



# Contact
stj@snu.ac.kr
