# Make a movie
style1 = "lines lt 4 lw 2"
style2 = "lines lt 3 lw 2"

range1 = "using 1:2"
range2 = "using 1:3"

system('mkdir -p animation')

do for [i=1:10]{
  filename = sprintf('data/solution-%03d.gnuplot',i)
  set terminal pngcairo dashed enhanced
  outfile = sprintf('animation/solution-%03d.png',i)
  set output outfile
  set multiplot layout 2,1
  plot filename @range1 with @style1
  plot filename @range2 with @style2
  unset multiplot
  reread}
