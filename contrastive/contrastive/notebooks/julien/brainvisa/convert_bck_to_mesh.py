import os
from soma import aims, aimsalgo

bcks_dir = '/neurospin/dico/jlaval/data/resampling/buckets/old/'
bcks_files = [elem for elem in os.listdir(bcks_dir) if elem[-1]!='f']
meshes_dir = '/neurospin/dico/jlaval/data/resampling/meshes/old/'

for bck_dir in bcks_files:
    bck = aims.read(bcks_dir + bck_dir)

    mesher = aimsalgo.Mesher()
    mesher.setVerbose(False)
    mesher.setDecimation(99.0, 3.0, 0.2, 180.0)
    mesher.setMinFacetNumber(50)
    mesh = aims.AimsSurfaceTriangle()
    # bck est le bucket a mailler
    mesher.getBrain(bck, mesh)
    aims.write(mesh, meshes_dir+bck_dir[:-4]+'.mesh')