import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Model 6h v3 (with OB Imbalance) ===\n")