import os
import requests

from mp_api.client import MPRester

MP_API_KEY = 'Xu1NqFMbAbfYegT6Sx4p5cy48KZEZxz8'
"""with MPRester(MP_API_KEY) as mpr:
    docs = mpr.summary.search(elements=["Si", "O"],
                              band_gap=(0.5, 1.0),
                              fields=["material_id",
                                      "formula_pretty",
                                      "composition", ])
for example_doc in docs:

    mpid = example_doc.material_id
    formula = example_doc.formula_pretty
    composition = example_doc.composition
    structure = mpr.get_structure_by_material_id(mpid)
    print(mpid, formula, composition, structure)"""
with MPRester(api_key="Xu1NqFMbAbfYegT6Sx4p5cy48KZEZxz8") as mpr:
    chemsys_formula = "Li-Fe-O"
    final_structures = mpr.get_structures(chemsys_formula, final=True)
print(final_structures)