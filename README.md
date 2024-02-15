# Reverse Image RAG  

### Synopsis: 
We build an API to retrieval-augment vision-language models with image and text context retrieved from the web.

Concretely, for a query image and query text (e.g. a question), we leverage reverse image search to find most similar images and their textual content / links.
The context is formatted an preprended to the prompt that finally is sent to the VLM (e.g. GPT4V).

The final product is a VLM-API that allows to automatically leverage reverse-image-search based retrieval augmentation.  

A visual overview with an example:  


![](img/slide1.png)

![](img/slide2.png)


### Usage:  


```python
api = RIR_API(openai_api_key)

image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"
query_text = "What is in this image?"
response = api.query_with_image(image_url, query_text)
# >> runs reverse image search
# >> formats image-text context prompt
# >> queries VLM with full query
```

