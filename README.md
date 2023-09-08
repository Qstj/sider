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

### Configurable parameters
```--epochs: how many epochs to train for each base model
--n_boost: how many base models
--split: split scheme for drug-side effect pairs. 'drug' split recommended
--drug_boost: if True, aggregates sample weights for Adaboost by drugs (recommended)
--copy_boost: if True, new base model is initialized to the weights of current training model (not recommended)
--lr: learning rate
--embed_dim: embedding dimensions for drug/side effect embeddings and vectors
--dropout: dropout rates for graph attention network and linear layers
--batch_size: how many drugs per batch
--mode: Adaboost classification or regression (classification recommended)
--folds: how many folds in cross validation
--seed: random seed
--session_name: session name for writing log files
--device: device to perform computations
--weight_decay: weight decaying penalty```


## NetGP
Drug target protein information vector is generated using NetGP, introduced in Pak et al. [1] We upload the codes of NetGP, to improve understanding of our model and to facilitate third-party application of our method.

```[1] Pak, Minwoo, et al. "Improved drug response prediction by drug target data integration via network-based profiling." Briefings in Bioinformatics 24.2 (2023): bbad034.```

### Using custom protein-protein interaction network

Prepare protein-protein interaction network formatted as:
```source	target	combined_score	
ARF5	PDE1C	155
ARF5	PAK2	197
ARF5	RAB36	222
ARF5	RAPGEF1	181
...
```


Load your own network using --ppi_fname parameter.

```python netgp.py \
    --datadir $WORKDIR/Data \
    --target_fname target_onehot_example.tsv \
    --outdir $WORKDIR/Results \
    --combined_score_cutoff 800 \
    --restart_prob 0.2 \
    --out_fname drug_target_profile.out \
    --ppi_fname 9606.protein.links.symbols.v11.5.txt
```



# Contact
stj@snu.ac.kr
