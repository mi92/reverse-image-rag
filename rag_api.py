""" Multimodal web-RAG API for image captioning and visual question answering with playwright"""

import os
import requests
#from serpapi import GoogleSearch
from openai import OpenAI
import json
import pickle 
import asyncio
from playwright.async_api import async_playwright

class RIR_API:
    """
    Reverse Image RAG API (RIR API) for image captioning and visual question answering.
    Steps:
    1. User provides an image URL and a query text.
    2. RIR API performs reverse image search and gets inline images with titles. (TODO: leverage result links for more context)
    3. RIR API queries a VLM (GPT4V) with the context of retrieved images and their titles, and the final query text.

    """
    def __init__(self, openai_api_key: str, 
                 debug=False, debug_cache:str='outputs'):
        """
        Initialize the RIR API with OpenAI API key.
        Inputs:
        - openai_api_key: (str) OpenAI API key
        """
        self.openai_api_key = openai_api_key 
        self.debug = debug
        self.debug_cache = debug_cache
        self.client = OpenAI(api_key=self.openai_api_key)

    def query_with_image(self, image_url, query_text=None):
        # Perform reverse image search and get inline images with titles
        inline_images_with_titles = self._reverse_image_search(image_url)
        
        # Query a VLM with the context of similar images and their titles
        response = self._query_vlm(inline_images_with_titles, image_url, query_text)
        return response

    
    def _run_search_by_image(self, image_url, k=5):
        # Handle the case where this is called from a synchronous context
        try:
            return asyncio.run(search_by_image(image_url, k))
        except RuntimeError as e:
            print(f'Error in reverse_image_search: {e}')
            # Handle the case where an event loop is already running
            # This is just an example and might not be the optimal way to handle this situation in a real app
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_by_image(image_url, k))

    def _reverse_image_search(self, image_url, k=5):
        """ run playwright-based image search and format results"""
        #loop = asyncio.get_event_loop()
        #if loop.is_running():
        #    # Handle the scenario where there's an existing event loop
        #    print("Loop is already running. Consider refactoring for direct async calls or proper event loop management.")
        #else:
        #    results = asyncio.run(search_by_image(image_url, k))
        results = self._run_search_by_image(image_url, k)
         
        # Format results: imageUrl --> image_url and Title --> text 
        results = [{'image_url': result['ImageUrl'], 'text': result['Title']} 
                   for result in results
        ]
        return results

    def _query_vlm(self, context_images_with_titles, query_image_url, query_text=None, max_tokens=200):
        # Initialize an empty list to hold all content items
        content_list = []

        # Here is context preamble:
        content_list.append({"type": "text", "text": 'Here is some context for the below query:'})

        # Add context:
        for images in context_images_with_titles:
            image_dict = {
                    "type": "image_url", 
                    "image_url": {"url": images['image_url']}, 
            }
            content_list.append(image_dict)
            content_list.append({"type": "text", "text": "Title: " + images['text']})

        # Add final query image
        content_list.append(
            {   "type": "image_url", 
                "image_url": {"url": query_image_url},
            }
        )
 
        # Add final query text (use default template if not available)
        if not query_text:
            query_text = "Describe this image."
        content_list.append({"type": "text", "text": "Query: " + query_text})

        print(f'Full query:')
        print(json.dumps(content_list, indent=4))

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
                messages=[
                    {"role": "user", "content": content_list}
                ],
            max_tokens=max_tokens,
        )
         
        if self.debug:
            out_folder = self.debug_cache
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
            with open(f'{out_folder}/cached_api_gpt4v_result.pkl', 'wb') as f:
                pickle.dump(response, f)
        return response

async def search_by_image(image_url, k=5):
        # Ensure k is positive and a reasonable number to avoid performance issues
        k = max(1, min(k, 20))  # Adjust the upper limit as needed
        results = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # Change to True for headless
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

            # Extract information from the first k search results
            results = await page.evaluate('''(k) => {
                const results = [];
                const containers = document.querySelectorAll('.G19kAf');
                for (let i = 0; i < containers.length; i++) {
                    const img = containers[i].querySelector('img');
                    // Log src for debugging
                    console.log(img ? img.src : 'No img found');

                    if (!img || img.src.startsWith('data:image/gif;base64')) {
                        // Log skipped placeholder
                        console.log('Skipped placeholder');
                        continue; // Skip placeholders
                    }

                    const imageUrl = img.src;
                    const aTag = containers[i].querySelector('a[role="link"]');
                    const title = aTag ? aTag.getAttribute('aria-label') : 'Title not found';

                    results.push({ imageUrl, title });
                    if (results.length >= k) break; // Stop after collecting enough valid results
                }
                return results;
            }''', k);
           
            await asyncio.sleep(5)  # Wait for 5 seconds

            print(f"Inside: {results}")
            
            #with open('tmp.pkl', 'wb') as f:
            #    pickle.dump(results, f)

            #await asyncio.sleep(5)  # Wait for 5 seconds
            
            #await browser.close()

        # return results 

        # print(f"Outside: {results}")
        # from IPython import embed; embed(); sys.exit()  
        # await browser.close()
        # Load from tmp file:
        #with open('tmp.pkl', 'rb') as f:
        #    results = pickle.load(f)
        #return results 


if __name__ == "__main__":
    # In case openai key in environment, use:
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_key = open('api_keys/openai_api_key.txt', 'r').read().strip()

    api = RIR_API(openai_api_key, debug=True)

    image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"

    query_text = "What is in this image?"
    response = api.query_with_image(image_url, query_text)

    print(response)


