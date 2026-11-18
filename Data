We use the Single-Family Loan-Level (SFLL) dataset provided by the Federal Home Loan Mortgage Corporation (FHLMC), commonly known as Freddie Mac, which contains loan-level data for a sizable share of mortgage loans in the United States.
URL: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Access and terms
----------------

Freddie Mac distributes the dataset after registration and sign-in, under the terms and
conditions linked from the page above (checked on 2026-11-18). The data are not in this
repository. Do not commit them to git.

Sample used in the paper
------------------------

Section 4 of the paper (arXiv version 2402.00299v2) describes the sample: loans originated in 2009
and 2010; monthly performance data from January 2012 to June 2013 for training and from July 2013
to December 2013 for testing; rolling windows of six monthly snapshots; and a target of 90 days or
more in arrears over the next 12 months. Borrowers whose zip codes share the first two digits are
in the same geographical area.

Files the notebooks expect
--------------------------

Every notebook reads and writes bare file names in its working directory.

- Data Processing/Data_Prep.ipynb reads one origination file and one monthly performance file in
  CSV format, named by the variables file_name_origdata_csv and file_name_MP_csv (not set in the
  notebook). The conversion of the raw Freddie Mac files to these CSV files is not in this
  repository.
- Data Processing/Data_Clean.ipynb writes df_orig_tr_sc.csv, df_orig_te_sc.csv, df_mp_tr_sc.csv
  and df_mp_te_sc.csv, which the notebooks in Network Construction read.
- The notebooks in Models read y_train.csv (column default) and G_area.gpickle. No notebook in this
  repository writes y_train.csv.

The README lists every step with its inputs and outputs.
