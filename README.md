# Using LLM for automatic node description generation for TAGs

Below is the instructions to run the code:

## Download datasets

Download and extract the dataset from the following link inside the `datasets` folder in the root directory:

[download link for dataset](https://uconn-my.sharepoint.com/:f:/g/personal/ldn24004_uconn_edu/EpVCrbZuoSBLsB-l_VMEs60BNfwekvrpPK9l9pZp8h76xw?e=bNNgLG)

## Run auroprompting pipeline

Run the autoprompt.sh file with the following arguments:

```
./autoprompt_seed.sh <seed> <dataset_name> <dataset_setting> <buffer_alpha> <node_texts>
```

Here, 
- `seed` is the seed value
- `dataset_name` is the name of dataset: [cora, pubmed, europe, usa, brazil]
- `dataset_setting`: `text_rich` for citation graphs, `text_free` for airport graphs
- `buffer_alpha`: alpha value for items retrieval from buffer using structural/textual embedding
- `node_text`: `title abstract` for citation graphs, `none` for airport graphs.