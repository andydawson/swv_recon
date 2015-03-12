
import graph;
import labelpath;

size(4inch,4inch,false);
defaultpen(fontsize(10pt));

real xmin = 0.0;
real xmax = 2.95;
real ymin = 0.0;
real ymax = 0.75;
real pad  = 0.1;
real h    = 0.75*pad;

path g;

/*
draw((xmin-pad,ymin-pad)
     --(xmax+pad,ymin-pad)
     --(xmax+pad,ymax+pad)
     --(xmin-pad,ymax+pad)--cycle, invisible);
*/

draw((xmin,ymin)
     --(xmax,ymin)
     --(xmax,ymax)
     --(xmin,ymax)--cycle);

real fcorr(real x) {
  x = 3.0 - x;
  return 1.0/(x * sqrt(2*pi)) * exp(-log(x)^2/2.0);
}

g = graph(fcorr, 0.0, 2.95);
draw(g);
labelpath(shift((-80,3))*Label("correlation of radial growth among trees and with climate"), g);


real fsens(real x) {
  x = 3.5 - x;
  return 1.0/(x * sqrt(2*pi)) * exp(-log(x)^2/2.0);
}

g = graph(fsens, 0.0, 2.95);
draw(g);

labelpath(shift((-80,-9))*Label("mean sensitivity"), g);

label("dense forest", (xmin+h,ymin-h/4), SE);
label("semiarid forest border", (xmax-h,ymin-h/4), SW);

//label("moisture limiting days", (xmin+h, ymin-2h), NE);
draw("moisture limiting days", (xmin+h,ymin-2h)--(xmax-h,ymin-2h), N, EndArrow);

label("low", (xmin-3h,ymin+h), S);
label("high", (xmin-3h,ymax-h), N);
draw((xmin-3*h,ymin+h)--(xmin-3*h,ymax-h), EndArrow);


draw("complacent", (xmin+h,ymax+h)--((xmin+xmax)/2,ymax+h), N, Arrows);
draw("sensitive", ((xmin+xmax)/2+h,ymax+h)--(xmax-6h,ymax+h), N, Arrows);


shipout('fritts', 'pdf');




