from __future__ import annotations

from scanpy import read, AnnData
from typing import Literal
from pathlib import Path

import scanpy as sc

zenodo_record_lt = "15334396"
zenodo_record_c2v = "22644169"

logg = sc.logging

__all__ = [
    "Liu_NSCLC_CD8",
    "Weinreb_in_vitro",
    "Erickson_murine_development",
]

def __dir__():
    return sorted(__all__)

def Liu_NSCLC_CD8(
    embedding_type: Literal["gex", "c2v"] = "gex",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from Liu et al. [PMID: 35121991] with CD8 T cells from NSCLC.

    Parameters
    ----------
    embedding_type : Literal["gex", "c2v"], optional
        Type of embedding to use, by default "gex".
    file_path : str | Path | None, optional
        Path where .h5ad-container will be stored, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset.
    """
    if embedding_type == "c2v":
        logg.info("using clonal embedding (compositions in adata.X)")
    elif embedding_type == "gex":
        logg.info("using gene expression embedding")
    else:
        raise ValueError("embedding_type must be one of 'gex' or 'c2v'")

    if file_path is None:
        file_path = f"data/Liu_CD8_{embedding_type}.h5ad"
        
    url = f"https://zenodo.org/records/{zenodo_record_c2v}/files/Liu_CD8_{embedding_type}.h5ad"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata

def Ireland_SCLC(
    embedding_type: Literal["gex", "c2v"] = "gex",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from Ireland et al. [PMID: 40963028] with murine SCLC organoid.

    Parameters
    ----------
    embedding_type : Literal["gex", "c2v"], optional
        Type of embedding to use, by default "gex".
    file_path : str | Path | None, optional
        Path where .h5ad-container will be stored, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset.
    """
    if embedding_type == "c2v":
        logg.info("using clonal embedding (compositions in adata.X)")
    elif embedding_type == "gex":
        logg.info("using gene expression embedding")
    else:
        raise ValueError("embedding_type must be one of 'gex' or 'c2v'")

    if file_path is None:
        file_path = f"data/Ireland_SCLC_{embedding_type}.h5ad"
        
    url = f"https://zenodo.org/records/{zenodo_record_c2v}/files/Ireland_SCLC_{embedding_type}.h5ad"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata

def Weinreb_in_vitro(
    embedding_type: Literal["gex", "c2v"] = "gex",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from Weinreb et al. [PMID: 31974159] with in vitro hematopoiesis.

    Parameters
    ----------
    embedding_type : Literal["gex", "c2v"], optional
        Type of embedding to use, by default "gex".
    file_path : str | Path | None, optional
        Path where .h5ad-container will be stored, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset.
    """
    if embedding_type == "c2v":
        logg.info("using clonal embedding (compositions in adata.X)")
    elif embedding_type == "gex":
        logg.info("using gene expression embedding")
    else:
        raise ValueError("embedding_type must be one of 'gex' or 'c2v'")
    
    if file_path is None:
        file_path = f"data/Weinreb_in_vitro_{embedding_type}.h5ad"
    
    url = f"https://zenodo.org/records/{zenodo_record_c2v}/files/Weinreb_in_vitro_{embedding_type}.h5ad"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata

def Erickson_murine_development(
    perturbed: bool = False,
    region: str = "trunk",
    subset: str = "all",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from [PMID: 40502176] with clonal atlas of murine development.

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