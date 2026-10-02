"""
linkfail  -  a tool that guesses whether a network link is HEALTHY or ABOUT TO FAIL.

THE BIG IDEA IN PLAIN WORDS
---------------------------
A long fibre-optic link is made of many short pieces called "spans". Each span has
an amplifier (boosts the light signal) and a stretch of fibre (weakens the signal).
Every span has an expected, "target" behaviour. When a link starts to go bad, its
real measurements drift away from those targets.

This tool:
  1. reads a table of measurements,
  2. measures "how far from normal" each link is (two numbers),
  3. learns from past examples which numbers mean "healthy" and which mean "failing",
  4. then labels new links.

HOW THE FILES FIT TOGETHER (think of a small factory line)
----------------------------------------------------------
  config.py       the instruction sheet: what settings exist and their defaults
  data.py         the preparation room: cleans the table and makes the two numbers
  models.py       the three "brains" (classifiers) that learn from examples
  evaluate.py     the exam marker: scores how good each brain is
  plotting.py     the artist: draws the pictures
  sample_data.py  the practice-data maker (fake but realistic measurements)
  cli.py          the control panel: the commands you type (train, predict, ...)
"""

# The version number. pyproject.toml reads it from here, so it only lives in one place.
__version__ = "1.0.0"
