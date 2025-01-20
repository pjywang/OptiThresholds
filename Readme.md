# Data-driven thresholds for optimal summary of wearable device data
A Python tool for identifying optimal, data-driven thresholds for summarizing wearable device data into Time-in-Range (TIR) proportions, which offers interpretable insights from high-frequency measurements. Two types of threshold optimality are considered: one tailored for data from a single population, and another preserves pairwise distribution distances. 

The detailed methods are described in the paper "**Beyond fixed thresholds: optimizing summaries of wearable device data via piecewise linearization of quantile functions**" by Junyoung Park, Neo Kok, and Irina Gaynanova. 

## Dependencies
`numpy`, `pandas`, and `scipy >= 1.12.0`.

## How to use
First import `method.py` file. A crucial preliminary step is to create a Python class `Distribution`, which is designed to handle univariate distributional data with empirical observations, such as data from wearable devices like accelerometers or CGMs. To prepare the data for creating a `Distribution` class, you need to convert it into a list of lists; each sublist represents an individual's empirical measurements, forming an empirical distribution.

### Processing example with CGM data

Let's say you have a processed CGM data saved in a `data.csv` file. This file includes columns `id`, `gl`, and `time`, representing the subject ID, measured glucose values, and the time of measurement, respectively. Then, you can use the following code to create a `Distribution` class :

```Python
import pandas as pd
from method import * # includes class Distribution

data = pd.read_csv("data.csv")
grouped_data = data.groupby('id').agg({'gl': list}).reset_index()

data_class = Distribution(grouped_data["gl"], ran=(40, 400))
```
Here, `ran=(40, 400)` specifies the measurement range of a typical CGM device, which should be specified in a data-dependent manner.

Using [Awesome-CGM](https://github.com/IrinaStatsLab/Awesome-CGM) or [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench) repositories, one can access publicly available CGM data and process it into .csv files as above.


### Data-driven thresholds using Distribution class
Once the data is converted to a `Distribution` class, you can apply the proposed algorithms DE, SA, and SS by using the `run_de`, `agglomerative_discrete`, and `divisive_discrete` functions, respectively. It is recommended to use `run_de` given its effective and efficient performance shown in the paper. Full example code is:

```Python
import pandas as pd
from method import * # includes class Distribution, run_de

data = pd.read_csv("data.csv")
grouped_data = data.groupby('id').agg({'gl': list}).reset_index()
data_class = Distribution(grouped_data["gl"], ran=(40, 400))

# Specify the target number of thresholds for summary
K = 4

# Optimality criteria: Loss1 preserves individual distributions & Loss2 preserves pairwise distances
loss = "Loss1"

# DE
best_cutoffs, min_loss = run_de(data_class, K=K, loss=loss)
```
`best_cutoffs` returns the optimal thresholds for the input data, and `min_loss` shows its achieved loss value.

### Semi-supervised implementation
If fixing some thresholds from domain knowledge is of interest (e.g., the time-in-range 70--180 mg/dL for CGM data), one can specify the `fixed` argument for this purpose. Here, one should specify `K` as the number of additional thresholds to search for their optimal positions.
```Python
# Find two additional thresholds besides fixed thresholds
K = 2

best_cutoffs, min_loss = run_de(data_class, K=K, loss=loss, fixed=(70, 181))
```


## Reproducing experiments in the paper
See `Simulations.ipynb` and `Real-data.ipynb` for details.


## Misc
- Paper will be posted on the ArXiv soon
- An R version of the proposed methods is in progress.
