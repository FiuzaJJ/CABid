import numpy as np

from .paths import SCOP_LOOKUP


def _codes(values):
    index = {}
    return np.array([index.setdefault(v, len(index)) for v in values], np.int32)


class Scop:
    """SCOPe40 lookup. Domains are referred to by their row in the table."""

    def __init__(self, path=SCOP_LOOKUP):
        rows = [line.rstrip("\n").split("\t")[:2]
                for line in open(path, encoding="utf-8") if line.strip()]
        self.ids = [r[0] for r in rows]
        sccs = [r[1] for r in rows]
        self.n = len(self.ids)

        sfam = [s.rsplit(".", 1)[0] for s in sccs]
        fold = [s.rsplit(".", 1)[0] for s in sfam]
        self.cls_name = np.array([s[0] for s in sccs])
        self.fam, self.sfam, self.fold = _codes(sccs), _codes(sfam), _codes(fold)
        self.n_fam = np.bincount(self.fam)[self.fam]
        self.n_sfam = np.bincount(self.sfam)[self.sfam]
        self.n_fold = np.bincount(self.fold)[self.fold]

        # Foldseek writes ids with a .pdb suffix, TMalign and DALI without
        self.key2idx = {}
        for i, d in enumerate(self.ids):
            self.key2idx[d] = i
            self.key2idx[d + ".pdb"] = i
