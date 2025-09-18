import numpy as np

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    energy = float(lines[1].strip())
    coords = []
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return energy, np.array(coords)
