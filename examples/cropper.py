import panel as pn

from hyperspec.registration import crop

pn.extension("bokeh")  # type: ignore

# NOTE: Don't bother passing the list of capture IDs if you just want to crop all of the captures in the directory
pn.serve(crop("../../data/set1", ["2023-03-09_006", "2023-03-09_004"]))
