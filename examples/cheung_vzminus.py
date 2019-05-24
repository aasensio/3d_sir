import sir3d


# Inverted model
iterator = sir3d.synth.Iterator(use_mpi=True, batch=256)

mod = sir3d.synth.Model('cheung_vzminus.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)
iterator.run_all_pixels()
