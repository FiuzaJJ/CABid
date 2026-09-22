# ranking -> (label, colour, linestyle)
STYLE = {
    "cabid":          ("CABid",                   "#62bc6b", "-"),
    "cabid_sym":      ("CABid symmetric",         "#2f8f57", "-"),
    "bitdensity":     ("bitdensity, q&t cov>=70", "#f989b9", "-"),
    "bitdensity_tgt": ("bitdensity, tcov>=70",    "#c76f9c", "-"),
    "bits":           ("Foldseek bits",           "#c8584f", "-"),
    "evalue":         ("Foldseek E-value",        "#fa9655", "--"),
    "foldseek_tm":    ("Foldseek-TM",             "#9b64c0", "-."),
    "tmalign":        ("TMalign",                 "#78a1ff", "-."),
    "dali":           ("DALI",                    "#009490", ":"),
}
ORDER = list(STYLE)

CLASS_LABEL = {"a": "a - all-alpha", "b": "b - all-beta", "c": "c - alpha/beta",
               "d": "d - alpha+beta", "f": "f - membrane", "g": "g - small"}


def label(key):
    return STYLE[key][0]


def colour(key):
    return STYLE[key][1]
