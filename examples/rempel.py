import sir3d

#iterator = sir3d.Iterator(use_mpi=False)

iterator = sir3d.synth.Iterator(use_mpi=True, batch=256, workers_slant=None)

mod = sir3d.synth.Model('rempel.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)

# iterator.run_all_pixels(rangex=[0,20], rangey=[0,20])
iterator.run_all_pixels()

