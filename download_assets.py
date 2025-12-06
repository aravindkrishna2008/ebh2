import os
import requests

assets = {
    "image2.png": "https://www.figma.com/api/mcp/asset/0e67ae77-4146-40a5-bb15-3fb2b28a7a79",
    "image3.png": "https://www.figma.com/api/mcp/asset/c5a9aa73-fc07-435e-a211-b8336bebc5b3",
    "rectangle7.png": "https://www.figma.com/api/mcp/asset/b9a1e6df-7df5-48a5-8140-b64b6cb15276",
    "frame.png": "https://www.figma.com/api/mcp/asset/8db975bd-596b-4f47-b38e-a6b3c634391c",
    "frame1.png": "https://www.figma.com/api/mcp/asset/c736bfe5-5dc1-44f5-8122-961eed69fb45",
    "frame2.png": "https://www.figma.com/api/mcp/asset/c7c6aa0f-2c63-4501-ac85-8ecdf6c4ed83",
    "line2.png": "https://www.figma.com/api/mcp/asset/b5772dee-8817-43d7-b99b-196f4339bcde"
}

output_dir = "/Users/archita/My project/ebh2/client/public/assets"

for name, url in assets.items():
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(os.path.join(output_dir, name), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
