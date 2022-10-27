import sys
import math


def demo_func(a=1):
    """
    Just a demo
    """
    print(f'this is a demo, a={a}')


# This will be executed when this module is imported for the first time
demo_func(17)


if __name__ == '__main__':
    """
    This will be executed when the script is run from the command line.
    """
    demo_func(a=42)

