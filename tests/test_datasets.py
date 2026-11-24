import pytest
from clone2vec import datasets


@pytest.fixture
def read_calls(monkeypatch):
    calls = []

    def fake_read(path, **kwargs):
        calls.append((path, kwargs))
        return "adata"

    monkeypatch.setattr(datasets, "read", fake_read)
    return calls


@pytest.mark.parametrize("embedding_type", ["gex", "c2v"])
@pytest.mark.parametrize("func, stem", [
    (datasets.Weinreb_in_vitro, "Weinreb_in_vitro"),
    (datasets.Lareau_in_vitro, "Lareau_in_vitro"),
    (datasets.Liu_NSCLC_CD8, "Liu_CD8"),
    (datasets.Ireland_SCLC, "Ireland_SCLC"),
])
def test_c2v_datasets_paths(read_calls, func, stem, embedding_type):
    assert func(embedding_type=embedding_type) == "adata"
    path, kwargs = read_calls[0]
    assert path == f"data/{stem}_{embedding_type}.h5ad"
    assert kwargs["backup_url"] == (
        f"https://zenodo.org/records/{datasets.zenodo_record_c2v}/files/{stem}_{embedding_type}.h5ad"
    )


@pytest.mark.parametrize("func", [
    datasets.Weinreb_in_vitro,
    datasets.Lareau_in_vitro,
    datasets.Liu_NSCLC_CD8,
    datasets.Ireland_SCLC,
])
def test_c2v_datasets_custom_path_and_invalid_type(read_calls, func):
    func(file_path="custom.h5ad")
    assert read_calls[0][0] == "custom.h5ad"
    with pytest.raises(ValueError):
        func(embedding_type="pca")


@pytest.mark.parametrize("perturbed, region, subset, fname", [
    (False, "trunk", "all", "Erickson_Trunk_Control_All.h5ad"),
    (True, "head", "neurons", "Erickson_Head_Perturb_Neurons.h5ad"),
    (False, "head", "mesenchyme", "Erickson_Head_Control_Mesenchyme.h5ad"),
])
def test_erickson_paths(read_calls, perturbed, region, subset, fname):
    datasets.Erickson_murine_development(perturbed=perturbed, region=region, subset=subset)
    path, kwargs = read_calls[0]
    assert path == f"data/{fname}"
    assert kwargs["backup_url"] == f"https://zenodo.org/records/{datasets.zenodo_record_lt}/files/{fname}"


def test_erickson_invalid_subset(read_calls):
    with pytest.raises(KeyError):
        datasets.Erickson_murine_development(subset="liver")
