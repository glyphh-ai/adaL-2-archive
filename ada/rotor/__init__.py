"""Rotors — sequenced memory pipelines over the Ada substrate.

A rotor runs steps in sequence against a prompt: WRITE the turn into
short-term memory, RECALL through the lattice + semantic layer to build
the prompt's fact tree, REASON the tree into a bounded just-in-time
injection, then route upstream. See docs/ROTOR.md.
"""
