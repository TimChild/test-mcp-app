"""Entrypoint when debugging (effectively the same as `reflex run`)."""

# import logging
# import os

from reflex.reflex import _run

# logging.basicConfig(level=logging.INFO)
# logging.critical(f"Starting with ENV: {os.getenv('ENV')}")

if __name__ == "__main__":
    _run()
