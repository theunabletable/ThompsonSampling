# -*- coding: utf-8 -*-
"""
Created on Sun May  4 19:50:25 2025

@author: Drew
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Decision:
    pid: str
    time: pd.Timestamp
    context: pd.Series
    action: Optional[int] = None
    p_send: Optional[float] = None
    reward: Optional[float] = None