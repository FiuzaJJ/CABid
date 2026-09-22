# CABid

Code for the CABid part of my master's thesis: length-normalised scores for
Foldseek alignments, and a pooled SCOPe40 benchmark to compare them with
Foldseek's own scores, TM-align, Foldseek-TM and DALI.

The toxins/ folder contains an HTML generated from an RMD file for the EVT estimation and the claude protein family mapping done

Protein structural alignment can detect homology where sequence similarity no longer has the power to do so. Foldseek made structural search orders of magnitude faster, enabling large many-against-many searches, but its scores are inherited from sequence alignment and accumulate over the alignment length. As a result, they are not comparable between queries, and any single threshold discards short alignments before low-quality ones. We propose two length-aware scores computed from values that Foldseek already reports, preserving its speed. Bitdensity normalizes the alignment score by the alignment length and uses a coverage filter on one of the proteins. Coverage-Aware Bitdensity (CABid) incorporates coverage as a smooth function, being able to rank all alignments, even those with low coverages. In a pooled SCOPe40 benchmark, both metrics excelled at recall at high precision, where they outperformed DALI, TM-align and Foldseek's native scores at both the family and the superfamily level. Symmetric CABid achieved the highest AUPRC of any method tested at the family level, while at the superfamily level it performed comparably to TM-align and slightly below Foldseek-TM.

Scores (bits = Foldseek structural bit score, spans in residues, coverages in %):

```
Bitdensity         bits / qspan,   only kept at the top of the list if qcov and tcov >= 70
CABid              bits / qspan^0.65 + 0.022599 * tcov
CABid symmetric    min(bits / qspan^0.65 + 0.022815 * tcov,
                       bits / tspan^0.65 + 0.022815 * qcov)
```

The exponent was chosen with a 5-fold cross-validation grouped by SCOPe fold,
and the coverage coefficients come from a logistic regression on superfamily
labels.

## Figures

```bash
pip install -r requirements.txt
python make_figures.py
```

This only reads `data/` and `cache/` (a few MB) and writes every figure below
to `figures/` as PNG and PDF.

| thesis | file |
|---|---|
| Figure 1 | `fig01_scope40_class_lengths` |
| Figure 5 | `fig05_self_alignment_evalue` |
| Figure 6 | `fig06_pr_curves_family` |
| Figure 7 | `fig07_benchmark_bars` |
| Figure 8 | `fig08_paired_forest` (also writes `significance_table.tsv`) |
| Figure 9 | `fig09_tp_before_nth_fp` |
| Figure 10 | `fig10_class_delta_r99` |
| Figure 11 | `fig11a_tmalign_fp_membrane`, `fig11b_tmalign_fp_small` (PyMOL, see `structures/`) |
| Figure 12 | `fig12a_tp_upset`, `fig12b_tp_scatter` |
| Figure S1 | `figS1a_cv_heldout`, `figS1b_cv_tp_before_nth_fp` |

Figure 11 is rendered with PyMOL: `python structures/build_domains.py` cuts the
four domains out of their PDB entries and `cd structures && pymol -cq fig11.pml`
draws the two panels.

## Rebuilding the cache

The raw inputs are the SCOPe40 all-against-all result tables from the Foldseek
benchmark (`foldseek.txt`, `foldseekTM.txt`, `TMalign.txt`, `dali.txt`, from
the foldseek-analysis data archive). They are around 7 GB and are not part of
this repository. Put them in `raw/` (or pass `--raw DIR`) and run, from the
repository root:

```bash
python -m scripts.extract_hits      # foldseek.txt -> cache/hit_columns, ~4 min
python -m scripts.pooled_curves     # PR curves, TPs before n-th FP, per class
python -m scripts.bootstrap         # B = 1000 paired bootstrap over queries
python -m scripts.tp_overlap        # TPs at 99% precision for Figure 12
python -m scripts.cv_sweep          # exponent sweep for Figure S1, ~20 min per exponent
python -m scripts.fit_kappa         # coverage coefficients at a = 0.65
```



## Benchmark

SCOPe40 has 11,211 domains, of which 3,566 can be evaluated (they have
relatives at family, superfamily and fold level). All alignments of all
queries go into one ranked list per score, instead of one list per query. The
labels and weights follow `bench.fdr.noselfhit.awk` from
[foldseek-analysis](https://github.com/steineggerlab/foldseek-analysis): a hit
to another fold is a false positive and each hit counts 1/n, where n is the
number of relatives the query has at that level. Each tool is scored on all
the alignments it reports.

## Layout

```
cabid/          scores, SCOPe lookup, pooled benchmark, logistic fit
scripts/        rebuild cache/ from the raw tables
make_figures.py thesis figures from cache/
data/           SCOPe40 lookup and domain lengths
cache/          precomputed results used by make_figures.py
structures/     domains and PyMOL script for Figure 11
figures/        output
```

## License

MIT, see `LICENSE`.
