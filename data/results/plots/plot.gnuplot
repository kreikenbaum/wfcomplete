# size 10
set title 'Classification Results for 10 Web Pages'
set term 'png' font '/usr/share/fonts/truetype/droid/DroidSans.ttf'
set output 'out.png'
set xrange [0:*]
set xlabel "overhead (in %)"
set yrange [0:100]
set ylabel "accuracy"
plot "disabled", "nocache", "cache", "a_i_noburst", "a_ii_noburst"#, "all" smooth csplines