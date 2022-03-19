# Plot 
style1 = "lines lt 4 lw 2"
style2 = "lines lt 3 lw 2"

range1 = "using 1:2"
range2 = "using 1:3"

do for [i=1:160]{
  set multiplot layout 2,1
  filename = sprintf('data/solution-%03d.gnuplot',i)
  plot filename @range with @style1
  plot filename @range2 with @style2
  unset multiplot
  pause 0.1}

pause mouse
