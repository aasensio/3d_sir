import sir3d

iterator = sir3d.synth.Iterator(use_mpi=False)

# iterator = sir3d.synth.Iterator(use_mpi=True, batch=100)

mod = sir3d.synth.Model('test.ini', rank=iterator.get_rank())
iterator.use_model(model=mod)

iterator.run_all_pixels(rangex=[0,30], rangey=[0,30])
# iterator.run_all_pixels()
