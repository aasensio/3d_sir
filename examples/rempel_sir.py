import sir3d


iterator = sir3d.synth.Iterator(use_mpi=True, batch=256)

mod = sir3d.synth.Model('rempel_sir.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)

iterator.run_all_pixels()

del iterator
del mod

iterator = sir3d.synth.Iterator(use_mpi=True, batch=256)

mod = sir3d.synth.Model('rempel_invert_sir.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)

iterator.run_all_pixels()
