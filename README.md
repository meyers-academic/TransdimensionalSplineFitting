# TransdimensionalSplineFitting
A small python package to do generic transdimensional spline fitting.

## To do:

* [ ] Make it pip installable
* [ ] Apply (uniform) priors on heights correctly (not sure they're implemented right now...but the example works)
* [ ] Add some custom jump proposals. Right now birth proposals draw from the prior. That won't work well in a lot of cases. An example of a good one to try is to take the nearest knot to the one that we're proposing to turn on and draw from a small Gaussian around that one. Example of how to implement something along these lines is in Section 2 here: https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-246X.2009.04226.x.
