"""Draw the thesis figures on CABid and the pooled SCOPe40 benchmark.

    python make_figures.py

Everything is read from data/ and cache/; the raw alignment tables are not
needed. Output goes to figures/ as PNG and PDF, named after the figure number
in the thesis.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from cabid import metric
from cabid.paths import BOOTSTRAP, CV, DOMAIN_LENGTHS, FIGURES, OVERLAP, POOLED, SELF
from cabid.scop import Scop
from cabid.style import CLASS_LABEL, ORDER, STYLE, colour, label

plt.rcParams.update({"savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "font.size": 9, "figure.facecolor": "white"})

LEVELS = ("family", "superfamily")
STATS = [("r99", "Recall at 99% precision"), ("auprc", "AUPRC")]
BOOTSTRAPPED = ["cabid_sym", "bitdensity", "cabid", "bitdensity_tgt", "bits",
                "foldseek_tm", "tmalign"]
HEAD = ["cabid_sym", "bitdensity", "cabid", "bitdensity_tgt", "bits", "evalue",
        "foldseek_tm", "tmalign"]
TOOL = {k: "foldseek" for k in ["cabid", "cabid_sym", "bitdensity", "bitdensity_tgt",
                                "bits", "evalue"]}
TOOL.update(foldseek_tm="foldseek_tm", tmalign="tmalign", dali="dali")

# The thesis text is 9.1 pt on a 440 pt wide page. Labels are scaled up so that,
# once a figure is shrunk to the page width, the smallest one is close to body
# size, but only as far as they do not start to overlap.
PAGE_W, BODY_PT = 440 / 72, 9.1


def enlarge_text(fig):
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    base = [t.get_fontsize() for t in texts]
    want = BODY_PT * fig.get_size_inches()[0] / PAGE_W / min(base)
    if want <= 1:
        return
    legends = [lg for lg in [ax.get_legend() for ax in fig.axes] + fig.legends if lg]
    in_legend = {id(t) for lg in legends for t in [*lg.get_texts(), lg.get_title()]}

    def problems(k, gap=1.0):
        for t, s in zip(texts, base):
            t.set_fontsize(s * k)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        g = gap * fig.dpi / 72
        boxes = [t.get_window_extent(r) for t in texts if id(t) not in in_legend]
        boxes += [lg.get_window_extent(r) for lg in legends]
        boxes = [Bbox.from_extents(b.x0 - g, b.y0 - g, b.x1 + g, b.y1 + g) for b in boxes]
        bad = {(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
               if boxes[i].overlaps(boxes[j])}
        for n, lg in enumerate(legends):
            host, b = (lg.axes or fig).bbox, lg.get_window_extent(r)
            if b.x0 < host.x0 - 1 or b.x1 > host.x1 + 1 or b.y0 < host.y0 - 1 or b.y1 > host.y1 + 1:
                bad.add(("legend", n))
        return bad

    already = problems(1.0, gap=3.0)
    lo, hi = 1.0, want
    if problems(hi) - already:
        for _ in range(6):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if problems(mid) - already else (mid, hi)
        problems(lo)


def save(fig, name, enlarge=True):
    if enlarge:
        enlarge_text(fig)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load_pooled():
    curves, stats, nth, cls = {}, {}, {}, {}
    for tool in sorted(set(TOOL.values())):
        z = np.load(POOLED / f"{tool}.npz")
        for k, v in zip(z["stat_keys"], z["stat_vals"]):
            name, lvl, s = str(k).split("|")
            stats[(name, lvl, s)] = float(v)
        for k, v in zip(z["nth_keys"], z["nth_vals"]):
            name, n, s = str(k).split("|")
            nth[(name, int(n), s)] = float(v)
        for k, v in zip(z["cls_keys"], z["cls_vals"]):
            cls[tuple(str(k).split("|"))] = float(v)
        for name in [k for k, t in TOOL.items() if t == tool]:
            for lvl in LEVELS:
                curves[(name, lvl)] = z[f"{name}|{lvl}|prec"], z[f"{name}|{lvl}|rec"]
    return curves, stats, nth, cls


# ---------------------------------------------------------------- Figure 1

def place_notes(fig, notes, bins, pad=0.03):
    """Put each statistics box in the top corner that needs the least headroom."""
    for _ in range(3):  # tick labels change width as the limits move
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax, note, line, counts, med in notes:
            bb = note.get_window_extent(r).transformed(ax.transAxes.inverted())
            w, h = bb.width + 2 * pad, bb.height + 2 * pad
            lo, hi = np.log10(ax.get_xlim())
            x = (np.log10(bins) - lo) / (hi - lo)
            m = (np.log10(med) - lo) / (hi - lo)
            need = {side: max(counts.max() * 1.03, under.max(initial=0) / (1 - h))
                    for side, under in (("left", counts[x[:-1] < w]),
                                        ("right", counts[x[1:] > 1 - w]))}
            side = min(need, key=need.get)
            note.set_position((pad if side == "left" else 1 - pad, 1 - pad))
            note.set_ha(side)
            hit = m < w if side == "left" else m > 1 - w
            line.set_ydata([0, 1 - h if hit else 1])
            ax.set_ylim(0, need[side])


def fig01_class_lengths():
    scop, L = Scop(), np.load(DOMAIN_LENGTHS)
    meaning = {"a": "All-α proteins", "b": "All-β proteins", "c": "α/β proteins",
               "d": "α+β proteins", "f": "Membrane/cell surface", "g": "Small proteins"}
    col = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868", "d": "#C44E52",
           "f": "#937860", "g": "#DA8BC3"}
    lengths = {c: L[(scop.cls_name == c) & (L > 0)] for c in meaning}
    allv = np.concatenate(list(lengths.values()))
    bins = np.logspace(np.log10(allv.min()) - 0.02, np.log10(allv.max()) + 0.02, 34)

    fig = plt.figure(figsize=(7.5, 7.4), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.7))
    fig.suptitle("SCOPe40 benchmark set: domain-length distribution per class", fontsize=14)
    notes = []
    for i, c in enumerate(meaning):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        v = lengths[c]
        counts, _, _ = ax.hist(v, bins=bins, color=col[c], edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        med = np.median(v)
        line = ax.axvline(med, color="0.25", ls="--", lw=1.0)
        note = ax.text(0.97, 0.97, f"n = {v.size:,}\nmedian {med:.0f} aa\n"
                       f"range {v.min()}–{v.max()}", transform=ax.transAxes,
                       ha="right", va="top", fontsize=9.5, color="0.35", linespacing=1.3)
        notes.append((ax, note, line, counts, med))
        ax.set_title(f"Class {c}\n{meaning[c]}", fontsize=12.5, pad=5)
        if i // 3 == 1:
            ax.set_xlabel("domain length (aa)", fontsize=12)
        if i % 3 == 0:
            ax.set_ylabel("domains", fontsize=12)
        ax.tick_params(labelsize=12, length=2.5)
        ax.locator_params(axis="y", nbins=4)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)

    place_notes(fig, notes, bins)

    ax = fig.add_subplot(gs[2, :])
    n = [lengths[c].size for c in meaning]
    ax.bar(list(meaning), n, color=[col[c] for c in meaning], edgecolor="white")
    for i, v in enumerate(n):
        ax.text(i, v + max(n) * 0.02, f"{v:,}", ha="center", fontsize=9.5, color="0.25")
    ax.set_ylim(0, max(n) * 1.18)
    ax.set_title("Domains per class", fontsize=12.5, pad=5)
    ax.set_xlabel("SCOPe class", fontsize=12)
    ax.set_ylabel("domains", fontsize=12)
    ax.tick_params(labelsize=12, length=2.5)
    ax.locator_params(axis="y", nbins=4)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig01_scope40_class_lengths", enlarge=False)


# ---------------------------------------------------------------- Figure 5

def fig05_self_alignments():
    s = pd.read_csv(SELF / "scope40_self_alignments.tsv", sep="\t")
    s = s[(s.alnlen < 150) & (s.evalue > 0)]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    h = ax.hist2d(s.alnlen, np.log10(s.evalue), bins=[60, 60], cmap="plasma",
                  norm=matplotlib.colors.LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, label="Count (log scale)")
    ax.set_xlabel("Alignment length (aa)")
    ax.set_ylabel("log$_{10}$(E-value)")
    ax.set_xlim(0, 150)
    ax.grid(False)
    ax.text(0.97, 0.03, f"n = {len(s):,}\nalnlen  median = {s.alnlen.median():.0f} aa\n"
            f"E-value median = {s.evalue.median():.2e}\nE-value min    = {s.evalue.min():.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.3", alpha=0.85))
    ax.set_title("Foldseek self-alignments: E-value vs Alignment length (alnlen < 150 aa)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig05_self_alignment_evalue")


# ---------------------------------------------------------------- Figure 6

def fig06_pr_curves(curves, stats, level="family"):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ORDER):
        for other in ORDER:
            if other != name:
                p, r = curves[(other, level)]
                ax.plot(r, p, color="0.82", lw=0.8, zorder=1)
        p, r = curves[(name, level)]
        lab, col, ls = STYLE[name]
        ax.plot(r, p, color=col, ls=ls, lw=2.2, zorder=3)
        ax.set_title(f"{lab}\nAUPRC {stats[(name, level, 'auprc')]:.3f}   "
                     f"R@99%P {stats[(name, level, 'recall@P99')]:.3f}", fontsize=8.5)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Recall (pooled)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    fig.tight_layout()
    save(fig, f"fig06_pr_curves_{level}")


# ---------------------------------------------------------------- Figure 7

def fig07_benchmark_bars(stats, ci):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4))
    xs = np.arange(len(ORDER))
    for i, (stat, slab) in enumerate(STATS):
        for j, lvl in enumerate(LEVELS):
            ax = axes[i][j]
            c = ci[(ci.level == lvl) & (ci.statistic == stat)].set_index("key")
            has_ci = np.array([k in c.index for k in ORDER])
            vals = np.array([c.loc[k, "point"] if k in c.index else
                             stats[(k, lvl, "auprc" if stat == "auprc" else "recall@P99")]
                             for k in ORDER])
            lo = np.array([c.loc[k, "ci_lo"] if k in c.index else np.nan for k in ORDER])
            hi = np.array([c.loc[k, "ci_hi"] if k in c.index else np.nan for k in ORDER])
            ax.bar(xs, vals, color=[colour(k) for k in ORDER], edgecolor="#55534e", lw=0.5)
            ax.errorbar(xs[has_ci], vals[has_ci],
                        yerr=[vals[has_ci] - lo[has_ci], hi[has_ci] - vals[has_ci]],
                        fmt="none", ecolor="#3d3b37", elinewidth=1.0, capsize=4)
            tops = np.where(has_ci, hi, vals)
            for x, v, t, ok in zip(xs, vals, tops, has_ci):
                ax.text(x, t + 0.015 * np.nanmax(tops), f"{v:.3f}", ha="center",
                        va="bottom", fontsize=7.5, color="black" if ok else "#777777")
            ax.set_xticks(xs)
            ax.set_xticklabels([label(k) for k in ORDER], rotation=30, ha="right", fontsize=8.5)
            ax.set_title(f"{lvl.upper()}  |  {slab}", fontsize=11)
            if j == 0:
                ax.set_ylabel(slab)
            ax.margins(y=0.18)
    fig.tight_layout()
    save(fig, "fig07_benchmark_bars")


# ---------------------------------------------------------------- Figure 8

def benjamini_hochberg(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank in range(p.size - 1, -1, -1):
        running = min(running, p[order[rank]] * p.size / (rank + 1))
        q[order[rank]] = running
    return q


def paired_tests(draws, subjects):
    """Each subject minus every other bootstrapped ranking, BH over all tests."""
    rows = []
    for s in subjects:
        for o in BOOTSTRAPPED:
            if o == s:
                continue
            for lvl in LEVELS:
                for stat, _ in STATS:
                    a, b = draws[f"{s}|{lvl}|{stat}"], draws[f"{o}|{lvl}|{stat}"]
                    d = a[1:] - b[1:]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / d.size)
                    rows.append(dict(subject=s, opponent=o, level=lvl, statistic=stat,
                                     delta=a[0] - b[0], ci_lo=lo, ci_hi=hi, p=p))
    t = pd.DataFrame(rows)
    t["q"] = benjamini_hochberg(t.p)
    t["significant"] = ((t.ci_lo > 0) | (t.ci_hi < 0)) & (t.q < 0.05)
    return t


def fig08_paired_forest(draws):
    subjects = {"cabid_sym": ("#2a78d6", "o"), "cabid": ("#1baf7a", "s"),
                "bitdensity": ("#eb6834", "^")}
    sig = paired_tests(draws, list(subjects))
    dodge = dict(zip(subjects, np.linspace(-0.24, 0.24, len(subjects))))
    panels = [(lvl, stat, slab) for lvl in LEVELS for stat, slab in reversed(STATS)]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), sharey=True)
    for ax, (lvl, stat, slab) in zip(axes.flat, panels):
        t = sig[(sig.level == lvl) & (sig.statistic == stat)].set_index(["subject", "opponent"])
        for y in range(0, len(BOOTSTRAPPED), 2):
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f3f0", lw=0, zorder=0)
        for y, other in enumerate(BOOTSTRAPPED):
            for s, (col, mk) in subjects.items():
                if s == other:
                    continue
                r = t.loc[(s, other)]
                yy = y + dodge[s]
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.8,
                        alpha=1 if r.significant else 0.45, solid_capstyle="round", zorder=2)
                ax.plot(r.delta, yy, marker=mk, ms=8.5 if mk == "^" else 7.5, ls="none",
                        mfc=col if r.significant else "white", mec=col, mew=1.6, zorder=3)
        ax.axvline(0, color="#2b2a28", lw=0.9)
        ax.set_ylim(len(BOOTSTRAPPED) - 0.5, -0.5)
        ax.set_yticks(range(len(BOOTSTRAPPED)))
        ax.set_yticklabels([label(k) for k in BOOTSTRAPPED], fontsize=10.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#d9d7d2", lw=0.6, alpha=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", colors="#6f6c66", labelsize=9.5)
        ax.set_title(f"{lvl.capitalize()}  ·  {slab}", fontsize=13, fontweight="semibold", pad=8)
    for j in range(2):
        s = sig[sig.statistic == panels[j][1]]
        lo, hi = min(s.ci_lo.min(), 0), max(s.ci_hi.max(), 0)
        for ax in axes[:, j]:
            ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
        axes[1, j].set_xlabel("Δ  =  reference  −  ranking on the row", fontsize=9.5)

    ref = [Line2D([], [], color=c, marker=m, ms=8, lw=1.8, label=label(s))
           for s, (c, m) in subjects.items()]
    sig_keys = [Line2D([], [], color="#55534e", marker="o", ms=8, lw=1.8,
                       label="95% CI excludes 0 and BH q < 0.05"),
                Line2D([], [], color="#55534e", marker="o", ms=8, lw=1.8, mfc="white",
                       alpha=0.6, label="not significant")]
    fig.legend(handles=ref, title="Reference ranking", loc="lower left",
               bbox_to_anchor=(0.12, 0), ncol=3, frameon=False, fontsize=10.5,
               title_fontsize=11, alignment="left")
    fig.legend(handles=sig_keys, title="Significance", loc="lower right",
               bbox_to_anchor=(0.97, 0), ncol=2, frameon=False, fontsize=10.5,
               title_fontsize=11, alignment="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.2, w_pad=2.0)
    save(fig, "fig08_paired_forest")
    sig.to_csv(FIGURES / "significance_table.tsv", sep="\t", index=False)


# ---------------------------------------------------------------- Figure 9

def fig09_tp_before_nth_fp(nth):
    nths = range(1, 7)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ends = []
    for name in HEAD:
        y = [nth[(name, n, "tp_total")] for n in nths]
        ax.plot(nths, y, color=colour(name), lw=1.7, marker="o", ms=8, mec="white",
                mew=0.8, label=label(name), zorder=3)
        ends.append((y[-1], colour(name)))
    ax.set_xticks(list(nths))
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th", "6th"])
    ax.set_xlabel("stop at the … false positive of the pooled ranking", fontsize=11)
    ax.set_ylabel("true positives retrieved before it", fontsize=11)
    ax.set_title("Pooled SCOPe40: TPs retrieved ahead of the n-th false positive", fontsize=14)
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(3500, 15000)
    ax.legend(fontsize=9, loc="lower left", frameon=False)

    # end labels, nudged apart where lines finish close together
    gap = 0.04 * np.diff(ax.get_ylim())[0]
    placed = []
    for y, col in sorted(ends):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(f"{int(y):,}", xy=(6, y), xytext=(6.12, y_lab), fontsize=10,
                    color=col, va="center", arrowprops=None if y_lab == y else
                    dict(arrowstyle="-", color=col, lw=0.6))
    fig.tight_layout()
    save(fig, "fig09_tp_before_nth_fp")


# ---------------------------------------------------------------- Figure 10

def fig10_class_delta(cls):
    others = [k for k in HEAD + ["dali"] if k != "cabid_sym"]
    classes = list(CLASS_LABEL)
    r99 = {k: np.array([cls[(k, c, "family", "recall@P99")] for c in classes])
           for k in others + ["cabid_sym"]}
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    xs = np.arange(len(classes))
    width = 0.8 / len(others)
    for i, k in enumerate(others):
        ax.bar(xs + (i - (len(others) - 1) / 2) * width, r99[k] - r99["cabid_sym"], width,
               color=colour(k), edgecolor="#55534e", lw=0.3, label=label(k))
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLASS_LABEL[c] for c in classes], rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Δ recall at 99% precision\n(other − CABid symmetric)", fontsize=11)
    ax.set_title("Family level, per class: CABid symmetric against each baseline", fontsize=11.5)
    ax.legend(fontsize=8, ncol=2, loc=(0.34, 0))
    fig.tight_layout()
    save(fig, "fig10_class_delta_r99")


# ---------------------------------------------------------------- Figure 12

TP_SETS = ["CABid symmetric", "bits", "bitdensity", "TMalign"]
TP_COL = {"CABid symmetric": "#1f9e3a", "bits": "#d62728", "bitdensity": "#882255",
          "TMalign": "#3b62c4"}


def fig12a_upset(z):
    rows = {n: z[f"rows|{n}"] for n in TP_SETS}
    member = np.zeros(int(z["n_hits"]), np.uint8)
    for i, n in enumerate(TP_SETS):
        member[rows[n]] |= np.uint8(1 << i)
    region = np.bincount(member, minlength=16)
    combos = sorted((c for c in range(1, 16) if region[c]), key=lambda c: -region[c])
    xs = np.arange(len(combos))
    counts = region[combos]
    sizes = [rows[n].size for n in TP_SETS]
    ys = np.arange(len(TP_SETS))

    fig = plt.figure(figsize=(5.8, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 3.15], width_ratios=[4.2, 1.0])
    axm = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1], sharey=axm)
    axb = fig.add_subplot(gs[1, 0], sharex=axm)

    for i, n in enumerate(TP_SETS):
        axm.scatter(xs, np.full(xs.size, i), s=74, lw=0,
                    c=[TP_COL[n] if c >> i & 1 else "#dddddd" for c in combos])
    for x, c in zip(xs, combos):
        on = [i for i in range(len(TP_SETS)) if c >> i & 1]
        if len(on) > 1:
            axm.plot([x, x], [min(on), max(on)], color="#888888", lw=1.2, zorder=0)
    axm.set_yticks(ys)
    axm.set_yticklabels(TP_SETS, fontsize=8.5)
    for i, t in enumerate(axm.get_yticklabels()):
        t.set_color(TP_COL[TP_SETS[i]])
    axm.set_ylim(len(TP_SETS) - 0.4, -0.6)
    axm.set_xlim(-0.6, len(combos) - 0.4)
    axm.tick_params(axis="x", bottom=False, labelbottom=False)
    axm.grid(False)
    axm.spines[["top", "right", "bottom"]].set_visible(False)

    axs.barh(ys, sizes, height=0.3, color=[TP_COL[n] for n in TP_SETS])
    for y, v in zip(ys, sizes):
        axs.text(v + 750, y, f"{v:,}", va="center", fontsize=8)
    axs.set_xlim(0, max(sizes) * 1.22)
    axs.set_title("set size", fontsize=9)
    axs.tick_params(axis="y", left=False, labelleft=False)
    axs.grid(axis="y", visible=False)
    axs.spines[["top", "right", "left"]].set_visible(False)

    axb.bar(xs, counts, color="#555555", width=0.72)
    for x, v in zip(xs, counts):
        axb.annotate(f"{v:,}", (x, v), xytext=(0, -3), textcoords="offset points",
                     ha="center", va="top", fontsize=9, rotation=90)
    axb.set_ylim(max(counts) * 1.16, 0)
    axb.set_ylabel("Superfamily TPs at 99% precision", fontsize=9.5)
    axb.tick_params(axis="x", bottom=False, labelbottom=False)
    axb.grid(axis="x", visible=False)
    axb.spines[["bottom", "right", "left"]].set_visible(False)

    fig.suptitle("UpSet plot\nTrue positives at 99% precision with each method", fontsize=12)
    save(fig, "fig12a_tp_upset")


def fig12b_tp_scatter(z, pts):
    rows = {n: z[f"rows|{n}"] for n in TP_SETS}
    where = {int(r): i for i, r in enumerate(pts["rows"])}

    def xy(r):
        i = np.fromiter((where[int(v)] for v in r), np.int64, len(r))
        return pts["tcov"][i], pts["qspan"][i]

    pairs = [("CABid symmetric", "bitdensity"), ("CABid symmetric", "bits"),
             ("CABid symmetric", "TMalign"), ("bitdensity", "TMalign")]
    fig, axes = plt.subplots(4, 1, figsize=(9, 18))
    all_x, all_y = [], []
    for ax, (A, B) in zip(axes, pairs):
        both = np.intersect1d(rows[A], rows[B], assume_unique=True)
        a_only = np.setdiff1d(rows[A], rows[B], assume_unique=True)
        b_only = np.setdiff1d(rows[B], rows[A], assume_unique=True)
        for r, col, lab, z_, mk, al in (
                (both, "#c8c8c8", f"both  (n={both.size:,})", 1, "o", 0.45),
                (a_only, TP_COL[A], f"{A} only  (n={a_only.size:,})", 2, "^", 0.85),
                (b_only, TP_COL[B], f"{B} only  (n={b_only.size:,})", 3, "v", 0.85)):
            x, y = xy(r)
            all_x.append(x)
            all_y.append(y)
            ax.scatter(x, y, s=7, c=col, alpha=al, marker=mk, lw=0, label=lab, zorder=z_)
        ax.set_yscale("log")
        ax.set_xlabel("target coverage (%)", fontsize=16)
        ax.set_ylabel("query span (residues, log)", fontsize=16)
        ax.tick_params(labelsize=15)
        ax.set_title(f"{A}  vs  {B}", fontsize=18, fontweight="semibold")
        ax.legend(fontsize=14.5, loc="upper left", markerscale=3.5, framealpha=0.9)
    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
    for ax in axes:
        ax.set_xlim(all_x.min() - 2, 101)
        ax.set_ylim(all_y.min() / 1.15, all_y.max() * 1.15)
    fig.tight_layout()
    save(fig, "fig12b_tp_scatter")


# ---------------------------------------------------------------- Figure S1

CV_MODELS = {"Xa": ("S/qspan^a", "#c8584f"),
             "Xa_ktcov": ("S/qspan^a + k*tcov   (CABid)", "#3b6bb5"),
             "Xa_cov70": ("S/qspan^a, cov<70% demoted", "#62bc6b"),
             "sym_max": ("max(S/qspan^a+k*tcov, S/tspan^a+k*qcov)", "#9b64c0")}
TM_COL = "#00b0b0"


def fold_mean(tab, model, lvl, stat):
    s = tab[(tab.model == model) & (tab.level == lvl)].groupby("a")[stat].mean()
    return s.index.to_numpy(float), s.to_numpy(float)


def a_star(train, model, lvl, stat):
    """Exponent chosen on the training folds."""
    a, v = fold_mean(train, model, lvl, stat)
    return a[np.nanargmax(v)]


def figS1_cv(stats, nth):
    train = pd.read_csv(CV / "cv_train_curves.tsv", sep="\t")
    test = pd.read_csv(CV / "cv_test_curves.tsv", sep="\t")
    tp_nth = pd.read_csv(CV / "cv_pooled_tp_nth.tsv", sep="\t")

    # (a) held-out rates
    ylab = {"auprc": "pooled AUPRC", "recall@P99": "recall @ 99% precision"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True)
    for i, stat in enumerate(ylab):
        for j, lvl in enumerate(LEVELS):
            ax = axes[i][j]
            for m, (lab, col) in CV_MODELS.items():
                a, v = fold_mean(test, m, lvl, stat)
                ax.plot(a, v, lw=1.9, color=col, label=lab)
                s = a_star(train, m, lvl, stat)
                ax.plot(s, np.interp(s, a, v), marker="*", ms=15, color=col, ls="none",
                        mec="white", mew=0.6, zorder=6)
            ax.axhline(stats[("tmalign", lvl, stat)], color=TM_COL, ls="--", lw=1.6,
                       label="TMalign (all 125.7M pairs)")
            ax.axvline(metric.A_EXP, color="#444444", lw=1.0, alpha=0.6)
            if i == 0:
                ax.set_title(lvl.upper())
            else:
                ax.set_xlabel("exponent a")
            ax.set_ylabel(ylab[stat])
    axes[0][0].legend(fontsize=7.2, loc="lower left")
    fig.suptitle("Grouped 5-fold CV over SCOPe folds. Held-out mean across folds.\n"
                 "Stars = a* selected on the training folds", fontsize=11)
    fig.tight_layout()
    save(fig, "figS1a_cv_heldout")

    # (b) pooled out-of-fold TPs ahead of the n-th FP
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.4), sharex=True)
    ordinal = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
    for ax, n in zip(axes.flat, range(1, 7)):
        for m, (lab, col) in CV_MODELS.items():
            s = tp_nth[(tp_nth.model == m) & (tp_nth.nth == n)].sort_values("a")
            ax.plot(s.a, s.tp_pooled, lw=1.7, color=col, label=lab)
            st = a_star(train, m, "family", "tp_before_1st_global_fp")
            ax.plot(st, np.interp(st, s.a, s.tp_pooled), marker="*", ms=13, color=col,
                    ls="none", mec="white", mew=0.6, zorder=6)
        ax.axhline(nth[("tmalign", n, "tp_total")], color=TM_COL, ls="--", lw=1.4)
        ax.axvline(metric.A_EXP, color="#444444", lw=1.0, alpha=0.6)
        ax.set_title(f"stop at the {ordinal[n - 1]} false positive", fontsize=9.5)
        if n in (1, 4):
            ax.set_ylabel("TPs before it")
    for ax in axes[1]:
        ax.set_xlabel("exponent a")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Pooled out-of-fold true positives ahead of the n-th false positive",
                 fontsize=11.5)
    fig.tight_layout()
    save(fig, "figS1b_cv_tp_before_nth_fp")


def main():
    FIGURES.mkdir(exist_ok=True)
    curves, stats, nth, cls = load_pooled()
    ci = pd.read_csv(BOOTSTRAP / "ci_b1000.tsv", sep="\t")
    draws = dict(np.load(BOOTSTRAP / "draws_b1000.npz"))
    overlap = np.load(OVERLAP / "tp_overlap_superfamily.npz")
    points = np.load(OVERLAP / "tp_points_superfamily.npz")

    fig01_class_lengths()
    fig05_self_alignments()
    fig06_pr_curves(curves, stats)
    fig07_benchmark_bars(stats, ci)
    fig08_paired_forest(draws)
    fig09_tp_before_nth_fp(nth)
    fig10_class_delta(cls)
    fig12a_upset(overlap)
    fig12b_tp_scatter(overlap, points)
    figS1_cv(stats, nth)


if __name__ == "__main__":
    main()
