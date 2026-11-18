# GAT-LSTM-ATT

Code for the paper:

> Sahab Zandi, Kamesh Korangi, María Óskarsdóttir, Christophe Mues, Cristián Bravo (2025). Attention-based dynamic multilayer graph neural networks for loan default prediction. *European Journal of Operational Research*, 321(2), 586–599. <https://doi.org/10.1016/j.ejor.2024.09.025>

Preprint: [arXiv:2402.00299](https://arxiv.org/abs/2402.00299).

The paper predicts one-year-ahead default on US mortgages from a dynamic multilayer network of
borrowers, built from Freddie Mac's Single-Family Loan-Level dataset. Two borrowers are connected
when they live in the same geographical area (one layer) or took their mortgage from the same
lending company (the other layer). A graph neural network (GCN or GAT) encodes each monthly
snapshot, a recurrent network (LSTM or GRU) links six consecutive snapshots, and an optional
attention layer weights the snapshots. This repository holds the notebooks for data preparation,
network construction and the eight GNN-RNN models. The notebooks are excerpts of the research code
and do not run top to bottom as published: see "What the notebooks do not contain" below.

Repository structure:

- `Data Processing`: notebooks for data preparation and cleaning.
- `Network Construction`: notebooks for building the network structures.
- `Models`: notebooks with the implementations of the models.
- `Data`: details about the dataset and a link to download it.

## Requirements

The notebooks were saved with a Python 3.7.6 kernel (notebook metadata). The package versions used
for the paper were not recorded.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages that the notebooks import. The first cell of each model
notebook also installs `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv`
from the PyTorch Geometric wheel index that matches the installed PyTorch and CUDA versions. The
network and model notebooks call `nx.write_gpickle` and `nx.read_gpickle`, which NetworkX 3.0
removed, so they need NetworkX 2.x.

Hardware: the arXiv version of the paper (Appendix C, Table C.2) reports one NVIDIA A100 GPU with
40 GB of memory and two CPU cores (AMD Milan 7413) per task. Tables 8 and 9 of the paper report the
training time of each model.

## Data

The data are Freddie Mac's Single-Family Loan-Level (SFLL) dataset:
<https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>. Freddie Mac distributes it
after registration and sign-in, under the terms and conditions linked from that page. The data are
not in this repository. The file `Data` in this repository describes the dataset, the sample used
in the paper and the files the notebooks expect.

Every notebook reads and writes bare file names in its working directory. There is no data
directory setting, so place the input files in the folder you run each notebook from.

## Training

There is no script. Open the notebooks in Jupyter and run them in this order.

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `Data Processing/Data_Prep.ipynb` | origination and monthly performance CSV files for one origination quarter (variables `file_name_origdata_csv` and `file_name_MP_csv`, not set in the notebook) | `df_orig_<year>_<quarter>.csv`, `df_mp_<year>_<quarter>.csv` |
| 2 | `Data Processing/Data_Clean.ipynb` | data frames `df`, `df_mp`, `df_train` and `df_test` and the list `ll` (the cells that load and split them are not in the notebook) | `df_orig_tot.csv`, `df_mp_tot.csv`, `df_orig_tr_sc.csv`, `df_orig_te_sc.csv`, `df_mp_tr_sc.csv`, `df_mp_te_sc.csv` |
| 3 | `Network Construction/Graph_Train.ipynb` | `df_orig_tr_sc.csv`, `df_mp_tr_sc.csv` | edge lists `area.csv` and `comp.csv`; graphs `G_area.gpickle` and `G_comp.gpickle` |
| 4 | `Network Construction/Graph_Test.ipynb` | `df_orig_te_sc.csv`, `df_mp_te_sc.csv` | edge lists `area_test.csv` and `comp_test.csv`; graphs `G_area_test.gpickle` and `G_comp_test.gpickle` |
| 5 | `Models/<model>.ipynb` | `y_train.csv` (column `default`), `G_area.gpickle` | no files; prints the loss and training accuracy every 100 epochs, and the training time |

The eight model notebooks cross two graph layers with two recurrent cells, with and without
attention:

| Notebook | Graph layer | Recurrent cell | Attention over snapshots |
|---|---|---|---|
| `Models/GCN_GRU.ipynb` | GCN | GRU | no |
| `Models/GCN_GRU_ATT.ipynb` | GCN | GRU | yes |
| `Models/GCN_LSTM.ipynb` | GCN | LSTM | no |
| `Models/GCN_LSTM_ATT.ipynb` | GCN | LSTM | yes |
| `Models/GAT_GRU.ipynb` | GAT | GRU | no |
| `Models/GAT_GRU_ATT.ipynb` | GAT | GRU | yes |
| `Models/GAT_LSTM.ipynb` | GAT | LSTM | no |
| `Models/GAT_LSTM_ATT.ipynb` | GAT | LSTM | yes |

Each model notebook removes the edges of a random half of the nodes, trains with binary
cross-entropy and Adam on windows of six snapshots, and leaves the hyperparameters (hidden sizes,
the number of attention heads in the GAT models, dropout, learning rate, number of epochs and
number of windows) as `--` placeholders. Appendix C (Table C.1) of the arXiv version lists 200
epochs, early stopping after 50, a learning rate of 0.001 and the Adam optimizer. The hidden sizes,
the number of heads and the dropout rate are not documented.

### What the notebooks do not contain

- The conversion of the raw Freddie Mac files to the CSV files that step 1 reads, and the helper
  function `f` and the variables `mergedDf`, `l1`, `year` and `quarter` that step 1 uses.
- The construction of the columns `area`, `provider` and `if_delq_sts`, of the target `default`
  and of `y_train.csv`, and the train and test split.
- The outlier capping and the min-max scaling described in Section 4.1 of the paper. Step 2 writes
  the `_sc` files without scaling.
- The loop over monthly snapshots. Steps 3 and 4 use an undefined month `s` and write one graph
  per layer; step 5 refers to `data1`, `data2` and so on, but builds only `data` from
  `G_area.gpickle`.
- The double layer network, the validation set and early stopping.
- In the model classes, the methods `_setup_layers` and `_base_*`, and, in the attention models,
  the attention weights `probs`.

## Evaluation

Not documented. The model notebooks import `AUROC` and `F1Score` from `torchmetrics` but do not
use them, and no notebook loads the test graphs of step 4. The code for the bootstrap confidence
intervals, the baselines (static GCN and GAT, logistic regression, XGBoost and a deep neural
network), the Shapley analysis and the attention scores is not in this repository.

## Pretrained models

Not released. The repository and its history contain no model files, the notebooks do not save
models, and the repository does not say why.

## Results

Table and figure numbers follow the arXiv version (2402.00299v2).

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 2 (network sizes) | Not documented | See Table 2 of the paper |
| Tables 3 and 4 (baseline models) | Not documented: the baseline code is not in this repository | See Tables 3 and 4 of the paper |
| Tables 5, 6 and 7 (dynamic models on the area layer, the company layer and the double layer) | Not documented: `Models/*.ipynb` train the models, the evaluation code is not in this repository | See Tables 5 to 7 of the paper |
| Tables 8 and 9 (training time) | Not documented: the model notebooks print the training time | See Tables 8 and 9 of the paper |
| Figures 7 and 8 (Shapley values) | Not documented | See Figures 7 and 8 of the paper |
| Figure 9 (attention scores) | Not documented | See Figure 9 of the paper |
| Tables A.1 and A.2 (descriptive statistics) | Not documented | See Appendix A of the paper |

## Citation

```bibtex
@article{zandi2025attention,
  title   = {Attention-based dynamic multilayer graph neural networks for loan default prediction},
  author  = {Sahab Zandi and Kamesh Korangi and María Óskarsdóttir and Christophe Mues and Cristián Bravo},
  year    = {2025},
  journal = {European Journal of Operational Research},
  volume  = {321},
  number  = {2},
  pages   = {586--599},
  doi     = {10.1016/j.ejor.2024.09.025},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

Apache-2.0. See `LICENSE`.

## Acknowledgements

The code was written by Sahab Zandi.

The paper's acknowledgements (arXiv version) name this support: the Natural Sciences and
Engineering Research Council of Canada (NSERC) through the Canada Graduate Scholarships – Doctoral
program (first author); the Economic and Social Research Council (ESRC), grant ES/P000673/1
(second and fourth authors); the Icelandic Research Fund, grant 228511-051 (third author); and
NSERC Discovery Grant RGPIN-2020-07114 (last author). The research was funded in part by the Canada
Research Chairs program, CRC-2018-00082. Compute Ontario, Calcul Québec and the Digital Research
Alliance of Canada provided computing support.
