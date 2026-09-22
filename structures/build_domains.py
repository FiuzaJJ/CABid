"""Cut the four SCOPe domains of Figure 11 out of their RCSB entries.

    python structures/build_domains.py

Domain boundaries are the SCOPe/ASTRAL ones (from the PDBe SIFTS SCOP mapping).
Model 1 only, protein atoms plus MSE, first alternate location.
"""

import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# domain, PDB entry, chain, first and last residue (None = whole chain)
DOMAINS = [("d1nkzb_", "1nkz", "B", 1, 41),
           ("d1q90m_", "1q90", "M", None, None),
           ("d1y02a2", "1y02", "A", 20, 70),
           ("d2gvia2", "2gvi", "A", 169, 201)]


def build(name, pdb, chain, first, last):
    url = f"https://files.rcsb.org/download/{pdb}.pdb"
    lines = urllib.request.urlopen(url).read().decode().splitlines(keepends=True)
    out, seen = [], set()
    for line in lines:
        rec = line[:6]
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM") or line[21] != chain:
            continue
        if rec == "HETATM" and line[17:20] != "MSE":
            continue
        resseq = int(line[22:26])
        if first is not None and not first <= resseq <= last:
            continue
        key = (line[12:16], resseq, line[26])
        if line[16] not in " A" or key in seen:
            continue
        seen.add(key)
        out.append("ATOM  " + line[6:16] + " " + line[17:])
    (HERE / f"{name}.pdb").write_text("".join(out) + "END\n")
    print(name, sum(l[12:16] == " CA " for l in out), "residues")


if __name__ == "__main__":
    for d in DOMAINS:
        build(*d)
