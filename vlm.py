import os
import base64
import torch
import time
from PIL import Image


class ModelHandler:
    """
    Handles interactions with various vision-language models.
    Supports both API-based models and local models.
    """
    
    def __init__(self, model_name, config):
        """
        Initialize the model handler.
        """
        self.model_name = model_name.lower()
        self.model_instance = None
        self.processor = None
        self.config = config

        self.hf_token = config["api_keys"]["huggingface"]
        self.hf_home = config["model_path"]["huggingface"]

        os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token
        os.environ["HF_HOME"] = self.hf_home
        
    def initialize_model(self):
        """
        Initialize the appropriate model based on the model name.
        Imports are done dynamically to avoid package conflicts.
        """

        # 第一步：把 qwen 加入到 API 组里，这样它就不会去本地找模型了
        if "gpt" in self.model_name or "claude" in self.model_name or "gemini" in self.model_name or "qwen" in self.model_name:
            print(f"Using {self.model_name} via API, no local model initialization required.")
            return # 这一行很重要，执行完就直接结束，不再往下跑报错的代码了

        elif "llama" in self.model_name:
            # Import Llama modules only when needed
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            
            if self.model_name == "llama-3.2-11b":
                print("Initializing Llama 3.2-11B model...")
                model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            elif self.model_name == "llama-3.2-90b":
                print("Initializing Llama 3.2-90B model...")
                model_path = "meta-llama/Llama-3.2-90B-Vision-Instruct"
            else:
                raise ValueError("Unsupported LLaMA model, please try again.")
                
            self.model_instance = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token
            )
            self.processor = AutoProcessor.from_pretrained(model_path, token=self.hf_token)
            
        elif "deepseek" in self.model_name:
            raise ValueError("Deepseek vision-language models are no longer supported.")
        
        else:
            raise ValueError("Unsupported model type, please try again")
    
    def get_response(self, prompt, image_path, max_tokens=500):
        """
        Get a response from the specified model and measures execution time.
        """
        start_time = time.time()
        response_text, token_counts = "", {"input": 0, "output": 0}
        
        if "gpt" in self.model_name.lower():
            response_text, token_counts = self._get_gpt_response(prompt, image_path, max_tokens, self.model_name)
            
        elif "claude" in self.model_name.lower():
            response_text, token_counts = self._get_claude_response(prompt, image_path, max_tokens, self.model_name)
            
        elif "gemini" in self.model_name.lower():
            response_text, token_counts = self._get_gemini_response(prompt, image_path, max_tokens, self.model_name)
            
        elif "qwen" in self.model_name.lower():
            response_text, token_counts = self._get_qwen_response(prompt, image_path)
            
        elif "llama" in self.model_name.lower():
            response_text, token_counts = self._get_llama_response(prompt, image_path)
            
        else:
            raise ValueError("Fail to get response, unsupported model type.")
            
        execution_time = time.time() - start_time
        return response_text, token_counts, execution_time
    
    def _get_gpt_response(self, prompt, image_path, max_tokens=500, model_name="gpt-4o-2024-11-20"):
        """
        Generate a response using GPT models via OpenAI API and count tokens.
        """
        try:
            import openai
            openai.api_key = self.config["api_keys"]["openai"]
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ]
            
            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
            )
            
            # Extract token counts from the usage field of the response
            token_counts = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            }
            
            return response.choices[0].message.content, token_counts
            
        except ImportError as e:
            print(f"OpenAI module not found: {e}")
            return "OpenAI module not available. Please install with: pip install openai", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in GPT inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}
        
    def _get_claude_response(self, prompt, image_path, max_tokens=500, model_name="claude-3-7-sonnet-20250219"):
        """
        Generate a response using Claude models via Anthropic API and count tokens.
        """
        try:
            import anthropic
            
            anthropic_api_key = self.config["api_keys"]["anthropic"]
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare message
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image", 
                        "source": {
                            "type": "base64", 
                            "media_type": "image/jpeg", 
                            "data": base64_image
                        }
                    }
                ]
            }
            
            # Get response from Anthropic
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[message]
            )
            
            # Extract token counts from the response
            token_counts = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
            
            return response.content[0].text, token_counts
            
        except ImportError as e:
            print(f"Anthropic module not found: {e}")
            return "Anthropic module not available. Please install with: pip install anthropic", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in Claude inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}

    def _get_gemini_response(self, prompt, image_path, max_tokens=500, model_name="gemini-2.5-pro-preview-03-25"):
        """
        Generate a response using Gemini models via Google GenAI API and count tokens.
        """
        try:
            import google.generativeai as genai
            
            gemini_api_key = self.config["api_keys"]["gemini"]
            
            # Configure the API with your key
            genai.configure(api_key=gemini_api_key)
            
            # Load the image
            image = Image.open(image_path)
            
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Create a chat session which is more reliable for vision tasks
            chat = model.start_chat(history=[])
            
            # Send the image and prompt together
            response = chat.send_message([
                prompt,
                image
            ])
            
            # Extract the response text
            response_text = response.text
            
            token_counts = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count
            }
            
            return response_text, token_counts

        except ImportError as e:
            print(f"Google GenAI module not found: {e}")
            return "Google GenAI module not available. Please install with: pip install google-generativeai", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in Gemini inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}

    def _get_qwen_response(self, prompt, image_path):
        """
        使用阿里云 DashScope API 调用千问模型。
        自动匹配命令行输入的模型名称，并修复 Windows 路径转义问题。
        """
        import dashscope
        from dashscope import MultiModalConversation
        import os
        from pathlib import Path
        import urllib.parse

        token_counts = {"input": 0, "output": 0}
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

        # 路径处理：修复 Windows 路径中的 + 号和反斜杠
        posix_path = Path(image_path).absolute().as_posix()
        encoded_path = urllib.parse.quote(posix_path, safe=':/')
        formatted_image_path = f"file://{encoded_path}"

        # 映射模型名称：将 LightEMMA 的简写映射为 API 要求的全称
        # 如果你开通的是 qwen-vl-max，请确保这里对应
        target_model = 'qwen-vl-max' 
        if "7b" in self.model_name:
            target_model = 'qwen2.5-vl-7b-instruct' # 或者是你开通的具体 7B 版本名称

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": formatted_image_path},
                    {"text": prompt}
                ]
            }
        ]

        try:
            # 开始 API 调用
            response = MultiModalConversation.call(model=target_model, messages=messages)
            
            if response.status_code == 200:
                output_text = response.output.choices[0].message.content[0]['text']
                if hasattr(response, 'usage'):
                    token_counts["input"] = response.usage.get("input_tokens", 0)
                    token_counts["output"] = response.usage.get("output_tokens", 0)
                return output_text, token_counts
            else:
                # 提示：如果依然报 Unpurchased，说明 target_model 指定的模型你没在后台开通
                return f"API Error: {response.code} - {response.message}", token_counts
                
        except Exception as e:
            return f"Error during Qwen API call: {str(e)}", token_counts
    def _get_llama_response(self, prompt, image_path):
        """
        Generate a response using the Llama 3.2 Vision model and count tokens.
        """
        token_counts = {"input": 0, "output": 0}
        
        try:
            if self.model_instance is None or self.processor is None:
                return "Llama model or processor not initialized. Please run in Llama environment.", token_counts
                
            # Import dependencies only when needed
            try:
                from PIL import Image
            except ImportError as e:
                return f"Required modules not available: {e}", token_counts
                
            # Load image as PIL Image
            image = Image.open(image_path).convert('RGB')
            
            # Format messages according to Llama's expected structure
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Get the model's primary device
            model_device = next(self.model_instance.parameters()).device
            
            # Process inputs and move directly to the model's device
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model_device)
            
            # Count input tokens
            token_counts["input"] = inputs.input_ids.shape[1]
            
            # Generate output
            with torch.no_grad():
                output = self.model_instance.generate(
                    **{k: v for k, v in inputs.items()},
                    max_new_tokens=512,
                    do_sample=False
                )
                
                # Calculate output token count
                token_counts["output"] = output.shape[1] - inputs.input_ids.shape[1]
            
            # Get the length of input tokens for decoding only new content
            input_length = inputs.input_ids.shape[1]
            
            # Decode only the new tokens (not the input tokens)
            output_text = self.processor.decode(output[0][input_length:], skip_special_tokens=True)
            
            # Extract the assistant's response using regex if needed
            import re
            assistant_response = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)
            
            if assistant_response:
                return assistant_response[0].strip(), token_counts
            else:
                return output_text.strip(), token_counts
            
        except Exception as e:
            print(f"Error in Llama inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}", token_counts