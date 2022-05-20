import os

import numpy as np
import pandas as pd
import math

from tsl import logger
from .prototypes import PandasDataset
from ..ops.similarities import gaussian_kernel
from ..utils import download_url, extract_zip


class Portland(PandasDataset):
    """
    This dataset contains 2 months
    (from 2011-09-15 to 2011-11-16) of 5 minutes traffic
    readings from 67 traffic detectors located on 6 Highways
    intersections of the I-205 Highway of the city of Portland,
    Oregon. This data was measured by the U.S. Department
    of Transportation, and offered on the PORTAL website
    (https://portal.its.pdx.edu/fhwa) as ”Test Data Set for the
    FHWA Connected Vehicle Initiative”. In order to build
    the adjacency matrix for this dataset, we have calculated
    the distances between each sensor’s coordinates using road
    distance (in Kilometers) utilizing the Google Maps Distance Matrix APIs,
    in order to re-create a precise and correct graph
    matrix between each sensor
    """
    url = "TBD"

    similarity_options = {'distance'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = None

    def __init__(self, root=None, impute_zeros=True, freq=None):
        # set root path
        self.root = root
        df, dist, mask = self.load()
        super().__init__(dataframe=df,
                         mask=mask,
                         attributes=dict(dist=dist),
                         freq="5T",
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="Portland")

    @property
    def raw_file_names(self):
        return ["portland.csv", "distances_portland.csv"]

    @property
    def required_file_names(self):
        return ["portland.csv", "distances_portland.csv"]

    def download(self) -> None:
        pass

    def build(self) -> None:
        pass

    def load_raw(self):
        pass

    def load(self, impute_zeros=True):
        # Loading CSV table
        df = pd.read_csv(
            "../data/portland/portland.csv", index_col=0)
        # Creating mask, utilizing the na values
        mask = df.notna().astype(int)
        # Filling the dataframe missing values with zeros
        df = df.fillna(0)
        # Updating the df index with DateTime Objects
        df.index = df.index = pd.DatetimeIndex(df.index, freq="5T")
        # Loading distance table
        dist = pd.read_csv(
            '../data/portland/distances_portland.csv', index_col=0)
        # Min Max Scaling to have distances between 0 and 1
        dist["dist"] = (dist["dist"]-dist["dist"].min()) / \
            (dist["dist"].max()-dist["dist"].min())

        # Create distance matrix, with un-connected nodes having -inf values
        dist = dist.pivot('detector_1', 'detector_2',
                          'dist').fillna(-1 * math.inf).values
        return df, dist, mask

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
