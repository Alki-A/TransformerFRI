# TransformerFRI
 Single-value classifications across functions of different protein given sequence-only data, using Transformer architecture.

 ## Prerequisites

Before running the project, ensure you have all dependencies installed:

1. **Snakemake**  
   
   Install dependencies (Snakemake is included here):
   ```bash
   pip install -r requirements.txt
   ```


 ## Usage

In order to run the correct pipeline, Snakemake can be used as follows:
   ```bash
   snakemake --cores <N>
``` 
where \<N\> is the number of CPU cores allocated.