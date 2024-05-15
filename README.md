# Small Molecule Drug Development Project

## Overview

This project is aimed at creating a classifcation predication model to identify small chemcial compounds that are potential inhibtiors of cancers caused by the BRAFV600E mutation, a mutation strongly assoicated with the onset of various forms of cancer, particulary melanoma. 

This program trains an SVM on the included file "chemical_compounds.csv", located locally relative to main.py in an 80/20 training split. After the training is completed, the program will print out an accuraccy and classfication report, demonstrating the model's effctiveness at predicting which small molecules are capable of inhbiting BRAF mutations. The program will then print out features identified by the model to be highly associated with the ability or not to inhibit BRAF mutations. Next, the model is then used to take in "new_chemical_compounds.csv" which is also located locally. The model then scans this csv and identifies molecules that may potentially inhibit the negative BRAF effects. After the identication of all molecules is completed, the results are stored in a local csv "predicted_braf_inhibitors.csv"

## Technical Details 

For this project, we have used an SVM (Support Vector Machine) classifier, which is useful for our project requirements where we have to categorize compounds based on the compounds likelihood of being an effective inhibitor for BRAF mutations.

## Resources 

* Pubchem 

## How to run 
First, ensure you have the neccesary requirements, listed below in the Requirements section. Run the following commands within the terminal in order to obtain the requirements

``` sh
pip3 install scikit-learn
pip3 install pandas
```

### Requirements 

* scikit-learn - Machine learning library
* pandas - Data analysis library












