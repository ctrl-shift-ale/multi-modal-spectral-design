"""
main.py

Entry point for the spectral design pipeline. Edit config.py, then run
this file (rather than priority_optimizer.py directly) -- this stays the
stable "run this" script even if the underlying optimizer implementation
changes later (e.g. when harmonic-level editing replaces the current
band-gain approach). priority_optimizer.py itself is still runnable
directly too; both do the same thing.
"""

from priority_optimizer import main

if __name__ == "__main__":
    main()
