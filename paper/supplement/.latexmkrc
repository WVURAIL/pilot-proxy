# Same convention as ../manuscript/.latexmkrc: with PP_OUT set, build
# products go to $PP_OUT/tex/supplement/ instead of the source tree.
if ($ENV{'PP_OUT'}) { $out_dir = "$ENV{'PP_OUT'}/tex/supplement"; }
