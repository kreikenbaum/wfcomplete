# size 10
set title 'Classification Results for 10 Web Pages'
set term 'png' font '/usr/share/fonts/truetype/droid/DroidSans.ttf'
set output 'out.png'
set xrange [0:*]
set xlabel "overhead (in %)"
set yrange [0:100]
set ylabel "accuracy"
rg = 10
f(x) = 90* exp(-x * a) + rg
a = 0.01
fit f(x) './all' via a
plot "disabled", "nocache", "cache", "a_i_noburst", "a_ii_noburst", "v0.19_all", "retrofixed", rg title "random guessing", "v0.20_ai", "v0.20_aii", "v0.20_bi", "v0.20_bii", f(x) title "exp(-x)"


#plot "disabled", "nocache", "cache", "a_i_noburst", "a_ii_noburst", "v0.19_ai@0", "v0.19_aii@0", "v0.19_bii@0", "v0.19_bi@20", "v0.19_bii@20"

# plot "disabled", "nocache", "cache", "a_i_noburst", "a_ii_noburst", "all" smooth csplines