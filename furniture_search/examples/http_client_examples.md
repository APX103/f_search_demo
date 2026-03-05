# JavaScript/Node.js 调用示例

```javascript
// Node.js + axios 示例
const axios = require('axios');
const fs = require('fs');

async function searchFurniture(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  const imageBase64 = imageBuffer.toString('base64');

  const response = await axios.post('http://localhost:8000/search', {
    image_base64: imageBase64,
    top_k: 20,
    category_hint: 'sofa'
  });

  console.log('Query description:', response.data.data.query_description);
  console.log('Results:', response.data.data.results);
}

searchFurniture('./design.jpg');
```

# Java 调用示例

```java
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.Base64;
import org.json.*;
import java.net.http.*;

public class FurnitureSearchClient {
    private static final String API_URL = "http://localhost:8000/search";

    public static String encodeImageToBase64(String imagePath) throws IOException {
        byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
        return Base64.getEncoder().encodeToString(imageBytes);
    }

    public static JSONObject searchFurniture(String imagePath) throws Exception {
        String imageBase64 = encodeImageToBase64(imagePath);

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("image_base64", imageBase64);
        requestBody.put("top_k", 20);
        requestBody.put("category_hint", "sofa");

        String jsonBody = new JSONObject(requestBody).toString();

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_URL))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        return new JSONObject(response.body());
    }

    public static void main(String[] args) throws Exception {
        JSONObject result = searchFurniture("design.jpg");
        System.out.println("Query description: " + 
            result.getJSONObject("data").getString("query_description"));
        System.out.println("Results: " + 
            result.getJSONObject("data").getJSONArray("results"));
    }
}
```

# Go 调用示例

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

type SearchRequest struct {
    ImageBase64  string `json:"image_base64"`
    TopK        int    `json:"top_k"`
    CategoryHint string `json:"category_hint,omitempty"`
}

type SearchResponse struct {
    Success bool `json:"success"`
    Data   struct {
        QueryDescription string `json:"query_description"`
        Results          []Result
    } `json:"data"`
    Meta struct {
        TookMs          int `json:"took_ms"`
        TotalCandidates int `json:"total_candidates"`
    } `json:"meta"`
}

type Result struct {
    ProductID      int     `json:"product_id"`
    SKU           string  `json:"sku"`
    Name          string  `json:"name"`
    Category      string  `json:"category"`
    Price         string  `json:"price"`
    Description   string  `json:"description"`
    LLMDescription string `json:"llm_description"`
    URL           string  `json:"url"`
    ImageURL      string  `json:"image_url"`
    Score         float64 `json:"score"`
    Rank          int     `json:"rank"`
}

func encodeImageToBase64(imagePath string) (string, error) {
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        return "", err
    }
    return base64.StdEncoding.EncodeToString(imageBytes), nil
}

func searchFurniture(imagePath string) (*SearchResponse, error) {
    imageBase64, err := encodeImageToBase64(imagePath)
    if err != nil {
        return nil, err
    }

    req := SearchRequest{
        ImageBase64:  imageBase64,
        TopK:        20,
        CategoryHint: "sofa",
    }

    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }

    resp, err := http.Post(
        "http://localhost:8000/search",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result SearchResponse
    err = json.NewDecoder(resp.Body).Decode(&result)
    if err != nil {
        return nil, err
    }

    return &result, nil
}

func main() {
    result, err := searchFurniture("design.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    fmt.Printf("Query description: %s\n", result.Data.QueryDescription)
    fmt.Printf("Found %d results in %dms\n", 
        result.Meta.TotalCandidates, 
        result.Meta.TookMs)
}
```

# Python 调用示例

```python
import requests
import base64
from pathlib import Path

def search_furniture(image_path: str) -> dict:
    """调用搜索服务"""
    # 读取并编码图片
    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    # 发送请求
    response = requests.post(
        "http://localhost:8000/search",
        json={
            "image_base64": image_base64,
            "top_k": 20,
            "category_hint": "sofa"
        }
    )
    
    return response.json()

# 使用示例
result = search_furniture("design.jpg")
print(f"Query: {result['data']['query_description']}")
for item in result['data']['results'][:5]:
    print(f"  - {item['name']} (score: {item['score']:.4f})")
```

# CURL 命令示例

```bash
# 读取图片并编码为 base64
IMAGE_BASE64=$(base64 -i design.jpg)

# 发送搜索请求
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_BASE64\",
    \"top_k\": 20,
    \"category_hint\": \"sofa\"
  }"

# 使用文件
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d @request.json
```

## API 响应格式

```json
{
  "success": true,
  "data": {
    "query_description": "现代简约风格的三人沙发，灰色布艺材质...",
    "results": [
      {
        "product_id": 12345,
        "sku": "SOFA-001",
        "name": "北欧简约三人沙发",
        "category": "沙发",
        "price": "¥2,999",
        "description": "商品描述...",
        "llm_description": "AI生成的描述...",
        "url": "https://...",
        "image_url": "https://...",
        "score": 0.0156,
        "rank": 1
      }
    ]
  },
  "meta": {
    "took_ms": 156,
    "total_candidates": 20
  }
}
```

## 错误处理

### 400 Bad Request
```json
{
  "detail": "Invalid base64 image data"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Search failed: Error message..."
}
```
