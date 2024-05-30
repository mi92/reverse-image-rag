# Reverse Image RAG - (RIR) 

> **[Paper code is on the `paper` branch]**
>
> The main branch of this repository hosts the code of the RIR package itself. 
> For the source code of our paper, [Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs](https://arxiv.org/abs/2405.18740), please visit the `paper` branch of this repository: [https://github.com/mi92/reverse-image-rag/tree/paper](https://github.com/mi92/reverse-image-rag/tree/paper).


![](img/ex1a.png)

![](img/ex1b.png)


### Synopsis: 
We build an API to retrieval-augment vision-language models with visual context retrieved from the web.

Concretely, for a query image and query text (e.g. a question), we leverage reverse image search to find most similar images and their titles / captions.

The final product is a VLM-API that allows to automatically leverage reverse-image-search based retrieval augmentation.  


### Usage:  

```pip install rir_api```

```python
import rir_api 

api = rir_api.RIR_API(openai_api_key)

image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"
query_text = "What is in this image?"
response = api.query_with_image(image_url, query_text)
# >> runs reverse image search
# >> formats visual context prompt
# >> queries VLM with full query
```

(see run.py for minimal example)

#### Debug mode:

For debugging, you can make API calls that display the web GUI (headless=True), and plot the image search result (show_result=True):   
```
response = api.query_with_image(image_url, query_text, show_result=True, delay=3, headless=False)

```

### Next steps  

- modularized API interface
- information extraction from search results 

Feel free to ping me under mdmoor[at]cs.stanford.edu if you're interested in contributing.

### Reference:  

@misc{Moor2024,  
  author = {Michael Moor},  
  title = {Reverse Image RAG~(RIR)},  
  year = {2024},  
  publisher = {GitHub},  
  journal = {GitHub Repository},  
  howpublished = {\url{https://github.com/mi92/reverse-image-rag}},   
}


