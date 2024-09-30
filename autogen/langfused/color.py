import os
import sys
from termcolor import colored


def safe_colored(text, color):
    # Disable colors if stdout is not a terminal (e.g., when running under Apache/mod_wsgi)
    if not os.isatty(sys.stdout.fileno()):
        return text  # Return plain text without color
    return colored(text, color)
