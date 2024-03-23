""" Multimodal web-RAG API for image captioning and visual question answering with playwright and screenshots"""

import os
import requests
from openai import OpenAI
import json
import pickle 
import asyncio
from playwright.async_api import async_playwright
import base64

class RIR_API:
    """
    Reverse Image RAG API (RIR API) for image captioning and visual question answering.
    Steps:
    1. User provides an image URL and a query text.
    2. RIR API performs reverse image search and gets inline images with titles. 
    3. RIR API queries a VLM (GPT4V) with the context of retrieved images and their titles, and the final query text.
    Step 3 is currently implemented via a screenshot of the search results as the image and text results are not returned consistently (yet).
    """

    def __init__(self, openai_api_key: str):
        """
        Initialize the RIR API with OpenAI API key.
        Inputs:
        - openai_api_key: (str) OpenAI API key,
        """
        self.openai_api_key = openai_api_key 
        self.client = OpenAI(api_key=self.openai_api_key)

    def query_with_image(self, 
                         image_url: str, 
                         query_text: str = None, 
                         output_path: str = None, 
                         delay: float = 2.,
                         show_result: bool = False,
                         headless: bool = True,
                         ):
        """
        Query the RIR API with an image URL and a query text.
        Inputs:
        - image_url: (str) URL of the image to query,
        - query_text: (str) query text to use in the VLM (GPT4V) API,
        - output_path: (str) path to save the API response as a pkl file.
        - delay: (float) delay in seconds to wait for the search results to load.
        - show_result: (bool) whether to plot the image search result screenshot.
        - headless: (bool) flag to deactivate browser gui to inspect search.
        """

        # Perform reverse image search and take a screenshot of the results
        screenshot_path = self._run_search_by_image(image_url, delay, headless)

        ## show the screenshot for few seconds if show_result flag is on:
        if show_result:
            from PIL import Image
            print(f"Showing the screenshot of the image search results to inspect the context.")
            img = Image.open(screenshot_path)
            img.show()

        # Encode the screenshot in base64 format
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the prompt for GPT-4V
        context_text = ("In the screenshot, the large image on the left is the query image for a reverse image search. "
                        "The smaller images on the right and their titles are the top hits from the search. "
                        "Please leverage any relevant context from the returned images and their titles in the following problem.")
        final_query_text = query_text or "Describe the following image:"

        # Construct the content list for the GPT-4V API
        content_list = [
            # Screenshot context:
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            # Text context / explanation
            {"type": "text", "text": context_text},
            # Query image
            {   "type": "image_url", 
                "image_url": {"url": image_url},
            },
            # Query text
            {"type": "text", "text": "Query: " + final_query_text},
        ]

        print(f'Querying GPT4V with augmented prompt...')

        # Call the GPT-4V API with the constructed content list
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": content_list}
            ],
            max_tokens=200,
        )

        if show_result:
            # close the img of the results again
            img.close()

        if output_path:
            # Write response to pkl:
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            with open(f'{output_path}/api_gpt4v_result.pkl', 'wb') as f:
                pickle.dump(response, f)

        return response

    def _run_search_by_image(self, image_url: str, delay: float = 2., headless=False):
        """ run playwright-based image search and return screenshot"""
        # Handle the case where this is called from a synchronous context
        try:
            return asyncio.run(search_by_image(image_url, delay=delay, headless=headless))
        except RuntimeError as e:
            print(f'Error in reverse_image_search: {e}')
            # Handle the case where an event loop is already running
            # This is just an example and might not be the optimal way to handle this situation in a real app
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_by_image(image_url, delay=delay, headless=headless))


async def search_by_image(image_url, screenshot_path='search_results.png', delay=2., headless=False):
    """
    Perform a reverse image search using the Playwright library and take a screenshot of the results.
    Inputs:
    - image_url: (str) URL of the image to search for,
    - screenshot_path: (str) path to save the screenshot.
    - delay: (float) delay in seconds to wait for the search results to load.
    - headless: bool to indicate if web search is done in headless mode (no gui browser opened)
    """
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)  # Change to True for headless
        ## page = await browser.new_page()
        context = await browser.new_context()  # Use a fresh context for each search
        page = await context.new_page()

        await page.goto('https://images.google.com')

        # Wait for the "Search by image" button to be visible
        await page.wait_for_selector('div[role="button"][aria-label="Search by image"]', state='visible')
        
        # Click on the "Search by image" button
        await page.click('div[role="button"][aria-label="Search by image"]')

         # Wait for the input field to be visible and fill it with the image URL
        await page.wait_for_selector('input[placeholder="Paste image link"]', state='visible')
        await page.fill('input[placeholder="Paste image link"]', image_url)

        # Click the "Search" button after entering the URL
        await page.wait_for_selector('div[jsname="ZtOxCb"]', state='visible')
        await page.click('div[jsname="ZtOxCb"]')
        # Further steps would go here, such as entering the image URL and submitting the search

        # Wait for the search results to load
        await page.wait_for_selector('img', state='visible')
      
        await asyncio.sleep(delay)  # Wait for few seconds for results to load 

       
        # Take a screenshot of the entire page
        await page.screenshot(path=screenshot_path, full_page=True) 

        await browser.close() ### Before used w/o close()

    return screenshot_path


if __name__ == "__main__":

    # In case openai key in environment, use:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    api = RIR_API(openai_api_key)
    
    # Example image:
    image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"

    query_text = "What is in this image?"
    # Regular API call:
    response = api.query_with_image(image_url, query_text, delay=3)

    # Debug API call that displays the web GUI, and plots the image search result: 
    # response = api.query_with_image(image_url, query_text, show_result=True, delay=3, headless=False)

    print(response)


