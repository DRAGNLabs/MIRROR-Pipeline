import sys

from lightning import Fabric

def rank_zero_log(fabric: Fabric, *args):
    if fabric.is_global_zero:
        # stderr because by default python buffers stdout, which can be confusing for debugging.
        # a lot of libraries seem to use this strategy of printing to stderr instead
        print(*args, file=sys.stderr)
