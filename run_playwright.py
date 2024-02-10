import asyncio
from playwright.async_api import async_playwright

async def search_by_image(image_url, k=5):
    # Ensure k is positive and a reasonable number to avoid performance issues
    k = max(1, min(k, 20))  # Adjust the upper limit as needed

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Change to True for headless
        page = await browser.new_page()
        await page.goto('https://images.google.com')

        # Wait for the "Search by image" button to be visible
        await page.wait_for_selector('div[role="button"][aria-label="Search by image"]', state='visible')
        
        # Click on the "Search by image" button
        await page.click('div[role="button"][aria-label="Search by image"]')

         # Wait for the input field to be visible and fill it with the image URL
        await page.wait_for_selector('input[placeholder="Paste image link"]', state='visible')
        await page.fill('input[placeholder="Paste image link"]', image_url)

        # Click the "Search" button after entering the URL
        await page.click('div[jsname="ZtOxCb"]')
        # Further steps would go here, such as entering the image URL and submitting the search

        # Wait for the search results to load
        await page.wait_for_selector('img', state='visible')

        # Extract the URLs and titles of the first few image results
        # Note: The actual selectors for images and titles may vary. This is a generic example.
        # Extract the URL and title of the first image result

        # Extract firt k image results:


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

        ## Good version with top k hits, but returns also placeholder urls
        #results = await page.evaluate(f'''() => {{
        #    const results = [];
        #    const containers = document.querySelectorAll('.G19kAf');
        #    for (let i = 0; i < Math.min(containers.length, {k}); i++) {{
        #        const container = containers[i];
        #        const img = container.querySelector('img');
        #        const imageUrl = img ? img.src : 'Image not found';

        #        const aTag = container.querySelector('a[role="link"]');
        #        const title = aTag ? aTag.getAttribute('aria-label') : 'Title not found';

        #        results.push({{ imageUrl, title }});
        #    }}
        #    return results;
        #}}''')
        print(results)

        # >> Worked with 1st hit: 
        ## Extract information from the first search result
        #result = await page.evaluate('''() => {
        #    // Target the container of the first result
        #    const container = document.querySelector('.G19kAf');
        #    if (!container) return { imageUrl: 'Container not found', title: 'Container not found' };

        #    // Extract the image URL
        #    const img = container.querySelector('img');
        #    const imageUrl = img ? img.src : 'Image not found';

        #    // Extract the descriptive text
        #    const aTag = container.querySelector('a[role="link"]');
        #    const title = aTag ? aTag.getAttribute('aria-label') : 'Title not found';

        #    return { imageUrl, title };
        #}''')
        #print(result)


        # For demonstration purposes, let's add a delay here to observe the action before closing the browser
        await asyncio.sleep(5)  # Wait for 5 seconds

        await browser.close()

image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"

asyncio.run(search_by_image(image_url))
