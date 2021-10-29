import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from modules.rruff_api import RruffApi

# -*- coding: utf-8 -*-

RruffApi = RruffApi()
RruffApi.run_main()


locs = RruffApi.locs

KMeans.fit(locs.to_numpy())