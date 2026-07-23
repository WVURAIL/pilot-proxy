# Project convention: when PP_OUT is set (the same env the analysis chain
# uses for every generated artifact), all LaTeX build products -- the
# compiled PDF and the aux files -- land under $PP_OUT/tex/manuscript/ and
# the source tree stays pristine. With PP_OUT unset (fresh clone, CI), the
# build is in-tree and .gitignore swallows it. No flags needed either way:
#   latexmk -pdf draft_article
if ($ENV{'PP_OUT'}) { $out_dir = "$ENV{'PP_OUT'}/tex/manuscript"; }
