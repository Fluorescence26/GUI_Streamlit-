from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from rdkit import Chem
from rdkit.Chem import Draw

import pandas as pd
idx = 0
compound_name = ""

def int_with_default(value, default):
    res = 0
    try:
        res = int(value)
    except ValueError:
        res = default
    return res

def search_compound():
    compound_row = df.loc[df['Smiles']==txt_compound.get()].to_numpy()[0]
    compound = compound_row[0]
    if (compound == ''):
        compound = txt_compound.get()

    img = Draw.MolToImage(Chem.MolFromSmiles(compound))
    img = ImageTk.PhotoImage(img)

    lbl_molecule.configure(image=img)
    lbl_molecule.image=img

    absorption_max = int_with_default(compound_row[2], -1)
    emission_max = int_with_default(compound_row[3], -1)
    if (absorption_max == -1 or emission_max == -1):
        lbl_status.configure(text="NOT Fluroscent")
    else:
        lbl_status.configure(text="Fluroscent")
    pass

df = pd.read_csv('All Properties with Finguprints_3.csv')

root = Tk()
root.geometry("800x600")

frm = ttk.Frame(root, padding=5)
frm.master.title("Fluroscence")

txt_compound = ttk.Entry(frm, textvariable=compound_name)
txt_compound.grid(row=0, column=0, columnspan=6)

btn_search = ttk.Button(frm, text="Search", command=search_compound)
btn_search.grid(row=0, column=6)

lbl_molecule = ttk.Label(frm, text="...")
lbl_molecule.grid(row=1, column=0, columnspan=4)

lbl_status = ttk.Label(frm, text="")
lbl_status.grid(row=2, column=0, columnspan=4)

frm.pack()
root.mainloop()
