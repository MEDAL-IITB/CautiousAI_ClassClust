# CautiousAI_ClassClust
Repo containing all the code of the CautiousAI Project : Cautious AI Improves Outcomes and Trust by Flagging Outlier Cases

Each of the four folders here contains the all the coode needed to run the complete set of experiments on that dataset.

**HAM**: This Folder contains all the code needed to run the experiments for HAM10000 dataset separated into 3 files by the OOD detection technique used. The same can be modofied to perform all class classification.

**KATHER**: This Folder contains all the code needed to run the experiments for KATHER colorectal separated into 3 files by the OOD detection technique used. The same can be modofied to perform all class classification.

**BACH**: The code files ending with "_\_bach.py_" are for running the base OOD experiments, Normal vs Invasive with InSitu as the OOD.  The code files ending with "_\_dbach.py_" are for the experiments with the damaged BACH (artificial deformities) dataset as OOD. The code files ending with "_\_bachcamelyon.py_" are for the experiments with the Breakhis, ETI and Camelyon dataset as OOD. The "_\_bach.py_" can be modified to run the 4 class classification experiments.

**MIAS**: In this folder files ending with "_\_mias\_new.py_" are the lates files on which we run our experiments. "_\_mias.py_" is an older version of the same code. The "_mias\_gen.py_" file is used tto preprocess the data from MIAS and Inbreast and generate the train , test and ood set folders used throughout experementation.

### NOTE: The code currently is in a experimental stage and will be made more presentable with time. Also, we will not be providing the dataset in this repo as all dataset we use are freely available for research on internet 
