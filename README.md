# Profiling Chemobiological Connection between Natural Product and Target Space Based on Systematic Analysis
This repository contains two relatively independent projects: 'data_analysis' and' FusionDTA_prediction '. Please ensure that they run in two different environments.

The core code is stored in the following six files:
1. data_analysis/code/1-data_preparing_lotus_NP.ipynb
2. data_analysis/code/2-NP_class_and_Taxon.ipynb
3. data_analysis/code/3-NP_activities.ipynb
4. data_analysis/code/4-hgih_active_NP.ipynb
5. FusionDTA_prediction/5-predicate_by_FusionDTA.ipynb
6. data_analysis/code/6-vusiual_prediction_result.ipynb

## Data Analysis
You can get the list of Natural product we make [here](https://zenodo.org/record/8047527)(iNP.db).

### 1-data_preparing_lotus_NP.ipynb: 

From the LOTUS database, the script extracts the lotus_id, SMILES, inchikey, and natural product classification information, including pathway, superclass, and class. Any missing classification data is retrieved from the NPClassfier API to ensure completeness.

### 2-NP_class_and_Taxon:

The following script retrieves taxonomic information such as kingdom and family for natural products from LOTUS mongoDB.

### 3-NP_activities.ipynb:


The script extracts data from the Table activities in the ChEMBL SQLite database, specifically retrieving only those molecules that have appeared in LOTUS.

### 4-hgih_active_NP.ipynb:

Statistical analysis of highly active natural products


### 6-vusiual_prediction_result.ipynb:

Statistical analysis of predicted affinity results between natural products and therapeutic targets

## DTA prediction
Firstly, download the ESM-1b model from [here](https://github.com/facebookresearch/esm), which we will use when generating protein representations.If you don't have enough graphics memory to generate protein representations, you can also download our prepared protein representations [here](https://zenodo.org/record/8047527)(task.pickle), and place it under "FusionDTA_prediction/" path.

Then click [here](https://drive.google.com/file/d/1FfFLPhM2-97qvgkzcTiU30PluRPCm6vU/view) to download the model we got from [FusionDTA](https://github.com/yuanweining/FusionDTA) and place it under "FusionDTA_prediction/pretrained_models/" path. 

Finally open "FusionDTA_prediction/5-predicateby_FusionDTA. ipynb" and run the code according to your own needs to get the DTA prediction result. You can also get our predicted results from [here](https://zenodo.org/record/8047527)(fusionDTA_task_result_1.bin) and use function "core.dec.read_result_bin" to load it.
