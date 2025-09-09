from src.utils.consts import CORA_ID2LABEL, PUBMED_ID2LABEL, AIRPORT_ID2LABEL

def label_name(dataset: str, y_id: int) -> str:
    if dataset == "cora":
        return CORA_ID2LABEL.get(int(y_id), f"unknown({y_id})")
    elif dataset == "pubmed":
        return PUBMED_ID2LABEL.get(int(y_id), f"unknown({y_id})")
    else:
        return AIRPORT_ID2LABEL.get(int(y_id), f"unknown({y_id})")