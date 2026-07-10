"""Offline box counting system for Raspberry Pi 5 + IMX219 camera.

Pipeline: capture -> background subtraction -> blob detection -> centroid
tracking -> directional line-crossing counting -> storage / GPIO / web UI.
"""

__version__ = "1.0.0"
