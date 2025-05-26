import boto3
import json
import os
import time
import logging
from random import uniform
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from paddleocr import TextRecognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Credentials and Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')  # Default to us-west-2 if not set
MODEL_CLAUDE_3_7 = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'#'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
MODEL_CLAUDE_3_5 = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def clean_json_string(s):
    """
    Clean JSON string by removing control characters and fixing common issues.
    """
    # Remove control characters
    clean = ''.join(char for char in s if ord(char) >= 32 or char in '\n\r\t')
    
    # Find the first { and last }
    start = clean.find('{')
    end = clean.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("No valid JSON object found in string")
    
    json_str = clean[start:end]
    
    # Fix common JSON formatting issues
    json_str = json_str.replace('\n', ' ')
    json_str = json_str.replace('\r', ' ')
    json_str = ' '.join(json_str.split())  # Normalize whitespace
    
    return json_str

def image_to_bytes(image):
    """Convert PIL Image to bytes and format for API"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return "jpeg", buffered.getvalue()


def extract_residence_card_info_with_ppocr(file_path):
    # 对示例图像执行 OCR 推理 
    result = ocr.predict(input=file_path)

    ocr_text = None
        
    # 可视化结果并保存 json 结果
    for res in result:
        # res.print()
        #print('res model_settings',json.dumps(res.get('rec_texts'), ensure_ascii=False))
        res.save_to_img("output")
        res.save_to_json("output")

        ocr_text = json.dumps(res.get('rec_texts'), ensure_ascii=False)

    return ocr_text
    

def extract_residence_card_info(image, ocr_text, max_retries=3):
    """
    Extract residence card information using Claude 3.7 Sonnet model with reasoning
    """
    try:
        logger.info("Using model: Claude 3.7 Sonnet with reasoning")
        
        # Resize image if needed
        max_size = 8000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)))
            logger.info(f"Image resized to {image.size}")
        
        # Convert image to bytes format
        file_type, image_bytes = image_to_bytes(image)
        
        # System message
        system_message = [{
            "text": "You are an expert at extracting information from residence cards and official identification documents."
        }]

        # User message with image
        user_message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": file_type,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                },
                {
                    "text": """Task: Extract key information from Japanese residence card images

                        Input: Image of a Japanese residence card (在留カード/zairyu card) which may be in any orientation.

                        Instructions:
                        1. First, determine if the image needs rotation and process accordingly.
                        2. Extract the json field information from the residence card.
                        3. Convert all dates to ISO format (YYYY-MM-DD) in the final output.
                        4. Follow the OCR recognition results to correct information.

                        OCR Results:
                        ###{ocr_text}###

                        Response Format:
                        Provide a valid JSON object with the following structure(Do not include any non-JSON information):
                        ```json
                        {
                        "name": {
                            "japanese": "Original Japanese name",
                            "roman": "Name in Roman letters"
                        },
                        "personal_info": {
                            "birth_date": {
                            "original": "Original date format",
                            "iso": "YYYY-MM-DD"
                            },
                            "gender": "Gender",
                            "nationality": "Nationality"
                        },
                        "address": "Complete address information",
                        "residence_details": {
                            "status": "Status of Residence type",
                            "card_number": "Card number",
                            "period": "Period of stay description",
                            "expiration_date": {
                            "original": "Original date format",
                            "iso": "YYYY-MM-DD"
                            },
                            "issue_date": {
                            "original": "Original date format", 
                            "iso": "YYYY-MM-DD"
                            },
                            "work_permission": "Work permission status"
                        },
                        "additional_info": {
                            "issuing_authority": "Issuing authority",
                            "other_details": "Any other important information"
                        }
                        }
                        """
                }
            ]
        }

        print('user_message: ', user_message)

        # Inference config
        inference_config = {
            "maxTokens": 4096,
            "temperature": 0
        }
        
        # Enable reasoning with a 2000 token budget
        reasoning_config = {
            "thinking": {
                "type": "disabled",
                "budget_tokens": 1024
            }
        }
        
        # API call with retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                if retry_count > 0:
                    delay = min(2 ** retry_count + uniform(0, 1), 5)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                
                logger.info("Calling Amazon Bedrock Converse API with Claude 3.7 Sonnet model and reasoning")
                response = bedrock_runtime.converse(
                    modelId=MODEL_CLAUDE_3_7,
                    messages=[user_message],
                    system=system_message,
                    inferenceConfig=inference_config,
                    #additionalModelRequestFields=reasoning_config
                )
                
                # Extract reasoning and final answer
                content_blocks = response["output"]["message"]["content"]
                
                reasoning = None
                output_text = None
                
                # Process each content block to find reasoning and response text
                for block in content_blocks:
                    if "reasoningContent" in block:
                        reasoning = block["reasoningContent"]["reasoningText"]["text"]
                    if "text" in block:
                        output_text = block["text"]
                
                if not output_text:
                    output_text = content_blocks[0]["text"]
                
                logger.info("Reasoning process captured")
                logger.info(f"\n<thinking>\n{reasoning}\n</thinking>")
                logger.info(f"Raw model response: {output_text}")
                
                # Clean and parse JSON
                try:
                    json_str = clean_json_string(output_text)
                    result = json.loads(json_str)
                    
                    return {
                        "status": "success",
                        "data": result,
                        #"raw_response": output_text,
                        "reasoning": reasoning
                    }
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse JSON: {str(e)}")
                    
                    if retry_count == max_retries - 1:
                        return {
                            "status": "error",
                            "message": f"Failed to parse response as JSON: {str(e)}",
                            #"raw_response": output_text,
                            "reasoning": reasoning
                        }
                    retry_count += 1
                    continue
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"API error: {error_msg}")
                
                if 'ThrottlingException' in error_msg and retry_count < max_retries - 1:
                    retry_count += 1
                    continue
                else:
                    return {
                        "status": "error",
                        "message": f"API error: {error_msg}"
                    }
        
        return {
            "status": "error",
            "message": f"Failed after {max_retries} retries"
        }
        
    except Exception as e:
        logger.exception(f"Error in residence card extraction: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def process_residence_card_images(directory_path="sample/"):
    """
    Process all residence card images in the specified directory and extract information
    """
    results = {}
    output_path = f"residence_card_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # List all image files in the directory
        image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
        # Initialize the JSON file with an empty object
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False)
        
        for image_file in image_files:
            file_path = os.path.join(directory_path, image_file)
            logger.info(f"Processing image: {file_path}")
            
            try:
                # extract info with ppocr
                ocr_text = extract_residence_card_info_with_ppocr(file_path)
                print("OCR result: " + ocr_text + "\n")

                # Open and process the image
                with Image.open(file_path) as img:
                    result = extract_residence_card_info(img, ocr_text=ocr_text)
                
                results[image_file] = result
                
                # Read existing content
                with open(output_path, 'r', encoding='utf-8') as f:
                    current_results = json.load(f)
                
                # Update with new result
                current_results[image_file] = result
                
                # Write back to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Finished processing {image_file} and saved to {output_path}\n")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                results[image_file] = {
                    "status": "error",
                    "message": f"Error processing image: {str(e)}"
                }
                
                # Read existing content
                with open(output_path, 'r', encoding='utf-8') as f:
                    current_results = json.load(f)
                
                # Update with error result
                current_results[image_file] = {
                    "status": "error",
                    "message": f"Error processing image: {str(e)}"
                }
                
                # Write back to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"All results saved to {output_path}")
        
        return results
    
    except Exception as e:
        logger.exception(f"Error in batch processing: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Process all residence card images in the sample directory
    results = process_residence_card_images("sample3/")
    print(f"Processed {len(results)} images")