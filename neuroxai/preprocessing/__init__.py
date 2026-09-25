"""Preprocessing of ADNI volumes into model inputs.

1. skull_strip: remove the skull with SynthStrip.
2. extract_slices: make 224x224 coronal PNG slices of the skull-stripped volumes.
"""
