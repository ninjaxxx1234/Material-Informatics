# We start by importing MPRester, which is available from the root import of pymatgen.
from pymatgen.ext.matproj import MPRester
from pprint import pprint
from pymatgen.ext.matproj import MPRester
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def get_material_data(api_key, element_pair):
    with MPRester(api_key) as mpr:
        material_ids = mpr.get_material_ids(element_pair)

        data = {
            "material_id": [],
            "symmetry": [],
            "composition": [],
            "formula_pretty": [],
            "energy_above_hull": [],
            "energy_per_atom": [],
            "formation_energy_per_atom": [],
            "nsites": [],
            "elements": [],
            "nelements": [],

        }

        for mpid in material_ids:
            with MPRester(api_key=api_key) as mpr:
                docs = mpr.summary.search(material_ids=[mpid],
                                          fields=["nsites", "elements", "nelements", "composition_reduced",
                                                  "formula_anonymous", "chemsys", "volume", "density",
                                                  "density_atomic", "property_name", "deprecated",
                                                  "deprecation_reasons", "last_updated", "origins", "warnings",
                                                  "structure", "task_ids", "uncorrected_energy_per_atom",
                                                  "is_stable", "equilibrium_reaction_energy_per_atom",
                                                  "decomposes_to", "xas", "grain_boundaries", "band_gap", "cbm",
                                                  "vbm", "efermi", "is_gap_direct", "is_metal", "es_source_calc_id",
                                                  "bandstructure", "dos", "dos_energy_up", "dos_energy_down",
                                                  "is_magnetic", "ordering", "total_magnetization",
                                                  "total_magnetization_normalized_vol",
                                                  "total_magnetization_normalized_formula_units",
                                                  "num_magnetic_sites", "num_unique_magnetic_sites",
                                                  "types_of_magnetic_species", "bulk_modulus", "shear_modulus",
                                                  "universal_anisotropy", "homogeneous_poisson", "e_total", "e_ionic",
                                                  "e_electronic", "n", "e_ij_max", "weighted_surface_energy_EV_PER_ANG2",
                                                  "weighted_surface_energy", "weighted_work_function", "surface_anisotropy",
                                                  "shape_factor", "has_reconstructed", "possible_species", "has_props",
                                                  "theoretical", "database_IDs", "symmetry""energy_above_hull", "composition", "symmetry", "formula_pretty",
                                                  "energy_per_atom", "formation_energy_per_atom"])

                data["material_id"].append(mpid)
                data["symmetry"].append(docs[0].symmetry)
                data["composition"].append(docs[0].composition)
                data["formula_pretty"].append(docs[0].formula_pretty)
                data["energy_above_hull"].append(docs[0].energy_above_hull)
                data["energy_per_atom"].append(docs[0].energy_per_atom)
                data["formation_energy_per_atom"].append(docs[0].formation_energy_per_atom)


                data["nsites"].append(docs[0].nsites)
                data["elements"].append(docs[0].elements)
                data["nelements"].append(docs[0].nelements)


        df = pd.DataFrame(data)
        return df


api_key = "Xu1NqFMbAbfYegT6Sx4p5cy48KZEZxz8" # you might want to change this
element_pair = "Ni-Cu"

df = get_material_data(api_key, element_pair)
print(df)
def plot_formation_energy_vs_composition(df):

    formation_energy = df["formation_energy_per_atom"]
    compositions = df["composition"]


    composition_labels = [str(comp) for comp in compositions]


    plt.figure(figsize=(10, 6))
    plt.scatter(composition_labels, formation_energy, s=50, alpha=0.7)


    for i, label in enumerate(composition_labels):
        plt.annotate(label, (composition_labels[i], formation_energy.iloc[i]), textcoords="offset points", xytext=(0, 5), ha='center')


    plt.xlabel("Composition")
    plt.ylabel("Formation Energy per Atom (eV)")
    plt.title("Formation Energy vs Composition Scatter Plot")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()


plot_formation_energy_vs_composition(df)

def search_materials_by_element_pair(api_key, element_pair):
    with MPRester(api_key) as mpr:
        material_ids = mpr.get_material_ids(element_pair)
        return material_ids

def get_material_information(api_key, material_id):
    with MPRester(api_key) as mpr:
        doc = mpr.get_doc(material_id)
        return doc

def get_material_summary(api_key, material_ids, fields):
    with MPRester(api_key) as mpr:
        docs = mpr.summary_docs(material_ids, fields=fields)
        return docs

def plot_formation_energyvscomposition(api_key, element_pair):
    df = get_material_data(api_key, element_pair)
    plot_formation_energy_vs_composition(df)
def get_material_properties(api_key, material_ids, properties):
    with MPRester(api_key) as mpr:
        data = mpr.query({"task_id": {"$in": material_ids}}, properties)
        return data

def calculate_derived_quantities(band_gaps):

    boltzmann_constant = 8.617333262145e-5  # eV/K, Boltzmann constant
    temperature = 300  # K, room temperature


    conductivity = np.exp(-band_gaps / (2 * boltzmann_constant * temperature))


    intrinsic_carrier_concentration = 2 * ((2 * np.pi * boltzmann_constant * temperature) / (6.62607015e-34))**1.5 * np.exp(-band_gaps / (2 * boltzmann_constant * temperature))


    effective_mass = (1 / (2 * np.pi))**2 * (np.gradient(band_gaps) / np.gradient(np.square(band_gaps)))

    return {
        "conductivity": conductivity,
        "intrinsic_carrier_concentration": intrinsic_carrier_concentration,
        "effective_mass": effective_mass
    }

def filter_materials_by_property(df, property_name, property_value):
    filtered_df = df[df[property_name] == property_value].reset_index(drop=True)
    return filtered_df


def get_bandstructure(api_key, material_ids):
    with MPRester(api_key) as mpr:
        bandstructures = mpr.get_bandstructure_for_material_ids(material_ids)

    data = {
        "material_id": material_ids,
        "bandstructure": bandstructures
    }

    df = pd.DataFrame(data)
    return df


def get_dos(api_key, material_ids):
    with MPRester(api_key) as mpr:
        dos_data = mpr.get_dos_for_material_ids(material_ids)

    data = {
        "material_id": material_ids,
        "dos": dos_data
    }

    df = pd.DataFrame(data)
    return df

def compare_materials(api_key, material_ids):
    details = [get_material_properties(api_key, m_id) for m_id in material_ids]

    # Extract band gaps for comparison
    band_gaps = [material['band_gap'] for material in details]

    # Simple comparison: Find the material with the maximum band gap
    max_band_gap = max(band_gaps)
    max_band_gap_material = details[band_gaps.index(max_band_gap)]


    return max_band_gap_material

def search_materials(api_key, keyword):
    with MPRester(api_key) as mpr:
        material_ids = mpr.search_materials({"text": keyword}, fields=["material_id"])
        return material_ids
def get_pure_element_data(api_key, chemsys):
    with MPRester(api_key) as mpr:
        elements = chemsys.split('-')  # Extract elements from chemsys

        data = {
            "material_id": [],
            "symmetry": [],
            "composition": [],
            "formula_pretty": [],
            "energy_above_hull": [],
            "energy_per_atom": [],
            "formation_energy_per_atom": []
            # Add other properties as needed
        }
        df = pd.DataFrame(data)

        for element in elements:
            element_composition = {element: 1}  # Pure element composition
            element_data = get_material_data(api_key, element_composition)
            df = df.append(element_data, ignore_index=True)

        return df

def export_to_csv(df, filename='materials_data.csv'):
    df.to_csv(filename, index=False)


plot_formation_energy_vs_composition(df)


plot_formation_energy_vs_composition(df)