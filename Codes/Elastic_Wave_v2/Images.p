# This file shows u(x,t) and u_t(x,t) through time

# Number of files
N = 160

#--- Finds the min and max of data ---#
stats 'data/solution-001.gnuplot' using 2 nooutput name 'Y1_'
stats 'data/solution-001.gnuplot' using 3 nooutput name 'Y2_'
do for [i=2:N]{
  stats sprintf('data/solution-%03d.gnuplot',i) using 2 nooutput name 'ytemp1_'
  stats sprintf('data/solution-%03d.gnuplot',i) using 3 nooutput name 'ytemp2_'
  if (ytemp1_min < Y1_min){Y1_min = ytemp1_min}
  if (ytemp1_max > Y1_max){Y1_max = ytemp1_max}
  if (ytemp2_min < Y2_min){Y2_min = ytemp2_min}
  if (ytemp2_max > Y2_max){Y2_max = ytemp2_max}
}
Y1_min = Y1_min-0.1*abs(Y1_min)
Y1_max = Y1_max+0.1*abs(Y1_max)
Y2_min = Y2_min-0.1*abs(Y2_min)
Y2_max = Y2_max+0.1*abs(Y2_max)

#--- Plot ---#
# Which data sets
range1 = "using 1:2"
range2 = "using 1:3"

# Line width and style
style1 = "lines lt 4 lw 2"
style2 = "lines lt 3 lw 2"

do for [i=1:50]{
  set multiplot layout 2,1
  filename = sprintf('data/solution-%03d.gnuplot',i)
  set yrange [Y1_min:Y1_max]
  plot filename @range with @style1 with @yrange1
  set yrange [Y2_min:Y2_max]
  plot filename @range2 with @style2
  unset multiplot
  pause 0.1}

pause mouse