try:
    from ._extension import compiled
except ImportError:

    def compiled(args):
        print("Warning: LASER extension not compiled, falling back to Python implementation.")
        return max(args, key=len)
