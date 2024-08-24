import boto3
import json

prompt_data = "Act as Shakespeare and write a poem on machine learning"

bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name="us-east-1"  # Ensure this matches your AWS Bedrock region
)

payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 50
}

body = json.dumps(payload)
model_id = "mistral.mistral-7b-instruct-v0:2"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_content = response.get("body").read()
    response_body = json.loads(response_content)
    
    # Extract and print only the poem text
    poem_text = response_body['outputs'][0]['text']
    print(poem_text)

except Exception as e:
    print(f"Error: {e}")
