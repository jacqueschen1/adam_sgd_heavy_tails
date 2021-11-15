def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))
