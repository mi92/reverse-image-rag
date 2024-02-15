import rir_api
import os

if __name__ == "__main__":

    openai_api_key = os.getenv("OPENAI_API_KEY")
    api = rir_api.RIR_API(openai_api_key)

    #image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY" # bird
    image_url = "http://tinyurl.com/2rrws56n" # fractal
    query_text = "What is in this image?"
    response = api.query_with_image(image_url, query_text)
    print(response)


