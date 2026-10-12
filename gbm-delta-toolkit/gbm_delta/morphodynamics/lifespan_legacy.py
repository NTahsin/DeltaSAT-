# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 03:17:33 2026

@author: tahsi001
# lifespan_calculator.py

this code uses rasterio for GeoTIFF handling and numpy for matrix operations.
"""


import os
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import List, Dict, Tuple

class LifespanCalculator:
    def __init__(self, classified_dir: str, output_dir: str, pixel_size_m: float = 30.0):
        """
        Initialize the LifespanCalculator.

        Args:
            classified_dir: Directory containing classified GeoTIFFs (one per year).
            output_dir: Directory to save transition matrices and results.
            pixel_size_m: Pixel size in meters (default: 30m for Landsat).
        """
        self.classified_dir = classified_dir
        self.output_dir = output_dir
        self.pixel_size_m = pixel_size_m
        os.makedirs(output_dir, exist_ok=True)

    def _load_geotiff(self, filepath: str) -> np.ndarray:
        """Load a GeoTIFF file as a numpy array."""
        with rasterio.open(filepath) as src:
            return src.read(1), src.profile

    def _compute_transition_matrix(self, year1: np.ndarray, year2: np.ndarray) -> np.ndarray:
        """
        Compute the transition matrix between two years.
        Classes: 0=land, 1=intertidal, 2=river.
        """
        # Flatten arrays and stack as columns
        stacked = np.vstack([year1.ravel(), year2.ravel()]).T
        # Count transitions
        transition_matrix = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                transition_matrix[i, j] = np.sum((stacked[:, 0] == i) & (stacked[:, 1] == j))
        return transition_matrix

    def _markov_lifespan(self, transition_matrix: np.ndarray) -> float:
        """
        Estimate the average lifespan of intertidal areas using Markov chains.
        Assumes intertidal (class 1) is the focus.
        """
        # Normalize rows to get transition probabilities
        P = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        # Fundamental matrix for transient states (intertidal)
        Q = P[1:2, 1:2]
        lifespan = 1 / (1 - Q[0, 0])  # Mean time in intertidal before absorption
        return lifespan

    def run(self, years: List[int]) -> Dict[int, float]:
        """
        Run lifespan calculation for all consecutive years.

        Args:
            years: List of years (e.g., [2000, 2001, 2002]).

        Returns:
            Dictionary: {year: lifespan_in_years}.
        """
        lifespans = {}
        for i in range(len(years) - 1):
            year1_path = os.path.join(self.classified_dir, f"intertidal_{years[i]}.tif")
            year2_path = os.path.join(self.classified_dir, f"intertidal_{years[i+1]}.tif")

            year1, _ = self._load_geotiff(year1_path)
            year2, _ = self._load_geotiff(year2_path)

            transition_matrix = self._compute_transition_matrix(year1, year2)
            lifespan = self._markov_lifespan(transition_matrix)
            lifespans[years[i]] = lifespan

            # Save transition matrix
            np.savetxt(
                os.path.join(self.output_dir, f"transition_{years[i]}_{years[i+1]}.csv"),
                transition_matrix,
                delimiter=",",
                fmt="%d",
            )

        return lifespans
