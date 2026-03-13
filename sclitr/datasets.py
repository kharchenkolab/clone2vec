from __future__ import annotations

from scanpy import read, AnnData
from pathlib import Path

zenodo_record_lt = "15334396"

def Weinreb_in_vitro(
    file_path: str | Path = "data/Weinreb_in_vitro.h5ad",
) -> AnnData:
    """
    Dataset from [PMID: 31974159] with in vitro hematopoiesis.

    Parameters
    ----------
    file_path : str | Path, optional
        Path where .h5ad-container will be stored, by default "data/Weinreb_in_vitro.h5ad".

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset.
    """
    url = f"https://zenodo.org/records/{zenodo_record_lt}/files/Weinreb_in_vitro.h5ad"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata

def Erickson_murine_development(
    perturbed: bool = False,
    region: str = "trunk",
    subset: str = "all",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from [PMID: None] with clonal atlas of murine development.

    Parameters
    ----------
    perturbed : bool, optional
        If the dataset should contain also experiments with mosaic knockouts,
        by default False.
    region : str, optional
        Region of interest, one of `trunk` or `head`, by default "trunk".
    subset : str, optional
        Subset of interest (`all`,  `neurons`, `mesenchyme` or `other`), by default "all".
    file_path : str | Path | None, optional
        Path where .h5ad-container will be stored, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset
    """
    datasets = {
        False: {
            "trunk": {
                "all": "Erickson_Trunk_Control_All.h5ad",
                "mesenchyme": "Erickson_Trunk_Control_Mesenchyme.h5ad",
                "neurons": "Erickson_Trunk_Control_Neurons.h5ad",
                "other": "Erickson_Trunk_Control_Other.h5ad",
            },
            "head": {
                "all": "Erickson_Head_Control_All.h5ad",
                "mesenchyme": "Erickson_Head_Control_Mesenchyme.h5ad",
                "neurons": "Erickson_Head_Control_Neurons.h5ad",
                "other": "Erickson_Head_Control_Other.h5ad",
            },
        },
        True: {
            "trunk": {
                "all": "Erickson_Trunk_Perturb_All.h5ad",
                "mesenchyme": "Erickson_Trunk_Perturb_Mesenchyme.h5ad",
                "neurons": "Erickson_Trunk_Perturb_Neurons.h5ad",
                "other": "Erickson_Trunk_Perturb_Other.h5ad",
            },
            "head": {
                "all": "Erickson_Head_Perturb_All.h5ad",
                "mesenchyme": "Erickson_Head_Perturb_Mesenchyme.h5ad",
                "neurons": "Erickson_Head_Perturb_Neurons.h5ad",
                "other": "Erickson_Head_Perturb_Other.h5ad",
            },
        },
    }
    
    dataset = datasets[perturbed][region][subset]
    
    if file_path is None:
        file_path = f"data/{dataset}"
        
    url = f"https://zenodo.org/records/{zenodo_record_lt}/files/{dataset}"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata