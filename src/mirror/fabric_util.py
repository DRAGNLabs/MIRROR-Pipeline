import sys

from lightning import Fabric

def rank_zero_log(fabric: Fabric, *args):
    if fabric.is_global_zero:
        print(*args, file=sys.stderr)
