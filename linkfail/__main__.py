"""
ROLE OF THIS FILE: lets you start the tool by typing  `python -m linkfail ...`

Python runs this file when you use `python -m linkfail`. All it does is hand over
to the real control panel (main() in cli.py).
"""

from .cli import main

# "__name__ == '__main__'" means "only do this when the file is run directly,
# not when another file merely imports it".
if __name__ == "__main__":
    main()
