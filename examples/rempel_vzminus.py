import sir3d

# Normal model
iterator = sir3d.synth.Iterator(use_mpi=True, batch=256)

mod = sir3d.synth.Model('rempel_vzminus.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)
iterator.run_all_pixels()

del iterator
del mod

# Inverted model
iterator = sir3d.synth.Iterator(use_mpi=True, batch=256)

mod = sir3d.synth.Model('rempel_vzminus_invert.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)
iterator.run_all_pixels()
