import numpy as np
from paper_xxiv_possibility_resolution_sweep_v2 import one
cells = [(40,0.01),(48,0.01),(40,0.015)]
NP, NS = 6, 6
for (k,cb) in cells:
    R = np.array([[one(k,cb,0.7,13,"random",ps,s)[0] for s in range(NS)] for ps in range(NP)])
    part_means = R.mean(axis=1)
    print(f"k={k} c_b={cb}: overall={R.mean():.2f}  across-partition std={part_means.std():.3f}  "
          f"within-partition seed std={R.std(axis=1).mean():.3f}  partmeans={np.round(part_means,2).tolist()}")
