# Contextual Distributionally Robust Optimization with Causal and Continuous Structure: An Interpretable and Tractable Approach

This library (CasualSDRO) is an open source project that is based on our paper:&#x20;

**_Contextual Distributionally Robust Optimization with Causal and Continuous Structure: An Interpretable and Tractable Approach_** (<https://arxiv.org/abs/2601.11016>).&#x20;

```latex
@article{zhang2026contextual,
  title={Contextual Distributionally Robust Optimization with Causal and Continuous Structure: An Interpretable and Tractable Approach},
  author={Zhang, Fenglin and Wang, Jie},
  journal={arXiv preprint arXiv:2601.11016},
  year={2026}
}
```

The experiments are coded in Python 3.8 and conducted on a personal computer equipped with an Intel Core i9-13900HX CPU, 32 GB of RAM, and an Nvidia GeForce RTX 4060 GPU. All GPU computations are performed using PyTorch 2.0.1 (utilizing CUDA 11.8).&#x20;

In the following, we introduce all folders and files, as well as their usage procedures.&#x20;

## 1. Main Programs for Solving DRO Problems

Here are all of the main programs in folders "DRO_Newsvendor", "DRO_Inventory", and "DRO_Portfolio"**.**

### 1.1 In Folder "**DRO_Newsvendor"**

This folder includes the following **runnable main programs**.&#x20;

| Files                                  | Descriptions                                                             |
| :------------------------------------- | :----------------------------------------------------------------------- |
| **\[1]Cross_Validation_Newsvendor.py** | Search best parameter combinations for all DRO models.                   |
| **\[2]Solver_Newsvendor.py**           | Solve Causal-SDRO Newsvendor with SRF and 2NN decision rules.            |
| **\[3]Compare_Newsvendor.py**          | Solve all DRO models with SRF decision rule.                             |
| **\[4]Visualization_Newsvendor.py**    | Visualize the performance of different decision rules or/and DRO models. |

### 1.2 In Folder "**DRO_Inventory"**

This folder includes the following **runnable main programs**.

| Files                                 | Descriptions                                                             |
| :------------------------------------ | :----------------------------------------------------------------------- |
| **\[1]Cross_Validation_Inventory.py** | Search best parameter combinations for all DRO models.                   |
| **\[2]Solver_Inventory.py**           | Solve Causal-SDRO Inventory with SRF and 2NN decision rules.             |
| **\[3]Compare_Inventory.py**          | Solve all DRO models with SRF decision rule.                             |
| **\[4]Visualization_Inventory.py**    | Visualize the performance of different decision rules or/and DRO models. |

### 1.3 In Folder "**DRO_Portfolio"**

This folder includes the following **runnable main programs**.

| Files                                           | Descriptions                                                                                       |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **\[Return and Features]20170101-20230331.csv** | All stock data from January 1, 2017 to March 31, 2023, including 5 features and 399 asset returns. |
| **\[1]Cross_Validation_Portfolio.py**           | Search best parameter combinations for all DRO models.                                             |
| **\[2]Solver_Portfolio.py**                     | Solve Causal-SDRO Portfolio with SRF and 2NN decision rules.                                       |
| **\[3]Compare_Portfolio.py**                    | Solve all DRO models with SRF decision rule.                                                       |
| **\[4]Visualization_Portfolio.py**              | Visualize the performance of different decision rules or/and DRO models.                           |

## 2. General Programs&#x20;

### 2.1 In Folder "**Tools**"

These programs will be called by the main programs.&#x20;

| Files                            | Descriptions                                             |
| :------------------------------- | :------------------------------------------------------- |
| **Data_Generator_Newsvendor.py** | Data generator for **newsvendor** problem.               |
| **Data_Generator_Inventory.py**  | Data generator for **inventory substitute** problem.     |
| **Decision_Rules.py**            | Two decision rules: SRF and 2NN.                         |
| **Functions.py**                 | Calculate losses and important metrics for all problems. |
| **Parameters.py**                | Fixed parameter classes for all problems.                |
| **Plotting.py**                  | Draw convergence curve and fitted distributions.         |

### 2.2 In Folder "**Optimizers**"

These programs will be called by the main programs.

| **Files**                     | Descriptions                                                                                |
| :---------------------------- | :------------------------------------------------------------------------------------------ |
| **ERM_Trainer.py**            | Solve ERM by SAA.                                                                           |
| **CSDRO_Trainer.py**          | Solve Causal-SDRO by SCSC.                                                                  |
| **SDRO_Trainer.py**           | Solve SDRO by SCSC.                                                                         |
| **CWDRO_Trainer.py**          | Solve Causal-WDRO by RT-MLMC (Yang et al., 2022).                                           |
| **KLDRO_Trainer.py**          | Solve KL-DRO by SGD.                                                                        |
| **Compared_DRO_Optimizer.py** | Solve all DRO models.                                                                       |
| **Portfolio_Optimizer.py**    | Solve all **portfolio** baselines and Causal-SDRO model.                                    |
| \*\*Gurobi_Optimizer.py \*\*  | Solve the inner problem of **inventory substitute problem** by Gurobi solver with gradient. |

### 2.3 In Folder "**Outputs**"

These files are generated by the main programs.

| Folders                    | Descriptions                                                                                                                                                                                                                                                          |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Outputs_Tuning**         | Record the best parameter combinations for all DRO models, generated by main programs named as \*\*\[1]Cross_Validation_Problem.py. \*\*This folder includes: DRO_Newsvendor_Best_Parameters.xlsxDRO_Inventory_Best_Parameters.xlsxDRO_Portfolio_Best_Parameters.xlsx |
| **Outputs_CSDRO**          | Record the results of Causal-SDRO model, generated by main programs named as \*\*\[2]Solver_Problem.py.\*\*This folder includes:DRO_Newsvendor_all_results.xlsxDRO_Inventory_all_results.xlsxDRO_Portfolio_all_results.xlsx                                           |
| **Outputs_DRO_Comparison** | Record the results of different DRO models, generated by main programs named as \*\*\[3]Compare_Problem.py.\*\*This folder includes:DRO_Compare_Newsvendor_results.xlsxDRO_Compare_Inventory_results.xlsxDRO_Compare_Portfolio_results.xlsx                           |
| **Outputs_Visualization**  | Visualize the performance of different decision rules or/and DRO models, generated by main programs named as \*\*\[4]Visualization_Problem.py. \*\*This folder includes three folders:Newsvendor_visualizationsInventory_visualizationsPortfolio_visualizations       |

## 3. Explanation

To show you how to use our codes to address the contextual DRO problem, we take the Newsvendor problem (in folder "DRO_Newsvendor") as an example.&#x20;

### 3.1 Tuning

Before solving the DRO models, please run the file **\[1]Cross_Validation_Newsvendor.py**

This function record the best parameter combinations for all DRO models, saving them to **Outputs_Tuning/DRO_Newsvendor_Best_Parameters.xlsx**.&#x20;

This table (**DRO_Newsvendor_Best_Parameters.xlsx**) is necessary, but running **\[1]Cross_Validation_Newsvendor.py** may take a long time! Thus, we have provided a finished **DRO_Newsvendor_Best_Parameters.xlsx** directly, and you don't need to run the cross validation procedure anymore.&#x20;

### 3.2 If you want to examine the SRF decision rule for Causal-SDRO.&#x20;

Please run **\[2]Solver_Newsvendor.py**, setting&#x20;

```Python
# Select decision rule(s)
USE_TwoLayerNN = False
USE_SRF = True
```

The results will be saved at **Outputs_CSDRO/DRO_Newsvendor_all_results.xlsx**.&#x20;

If you want to check convergence figures and fitted distributions, please set

```Python
# Output figures
Convergence_figure = True
Distribution_figures = True
```

and the figures will be saved at the folder **Outputs_CSDRO.**

### 3.3 If you want to compare the SRF and 2NN decision rules for Causal-SDRO.

Please run **\[2]Solver_Newsvendor.py**, setting

    # Select decision rule(s)
    USE_TwoLayerNN = True
    USE_SRF = True

The results will be saved at **Outputs_CSDRO/DRO_Newsvendor_all_results.xlsx**.

If you want to check convergence figures and fitted distributions, please set

    # Output figures
    Convergence_figure = True
    Distribution_figures = True

and the figures will be saved at the folder **Outputs_CSDRO.**

### 3.4 If you want to compare Causal-SDRO with other DRO models.

Please run **\[3]Compare_Newsvendor.py**.&#x20;

The results will be saved at **Outputs_DRO_Comparison/DRO_Compare_Newsvendor_results.xlsx.**

### 3.5 If you want to visualize the performance of different decision rules or/and DRO models.

Please run **\[4]Visualization_Newsvendor.py**, setting

```Python
# Visualize the performance of different decision rules.
show_results = True
# Visualize the performance of different DRO models.
show_dro_comparison = True
```

Ensure that the results have been saved at the corresponding folders.&#x20;

The resulting boxplots will be saved at folder **Outputs_Visualization/Newsvendor_visualizations**.&#x20;

## Reference&#x20;

Yang, J., Zhang, L., Chen, N., Gao, R., & Hu, M. (2022). Decision-making with side information: A causal transport robust approach.Ã‚Â *Optimization Online*.
