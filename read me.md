## Overview

CellAL mainly consists of two procedures: LRI prediction by feature extraction via BioTriangle, feature selection through XGBoost, and classification using LSTM network with the attention mechanism, along with CCC inference based on LRI filtering, CCC inference and CCC network visualization.

![Overview](C:\Users\14177\Desktop\Overview.png)

## Data

1.Data is available at [uniprot](https://www.uniprot.org/), [GEO](https://www.ncbi.nlm.nih.gov/geo/).

2.Feature extraction website at [BioTriangle](http://biotriangle.scbdd.com/)

## Usage



1. We obtain ligand and receptor feature at [BioTriangle](http://biotriangle.scbdd.com/)

2. Run the model to obtain the LRI, or the user-specified LRI database

   ```
   python code/main.py
   ```

## Cell-cell communication tools for comparative analysis



[CellPhoneDB](https://github.com/Teichlab/cellphonedb)  [iTALK](https://github.com/Coolgenome/iTALK)   [CellChat](https://github.com/sqjin/CellChat) 