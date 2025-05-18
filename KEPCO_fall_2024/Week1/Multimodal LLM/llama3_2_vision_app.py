import gradio as gr
import torch
import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, GenerationConfig

# unsloth/Llama-3.2-11B-Vision-bnb-4bit
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/workspace/nmquy/hf_cache"
)
processor = AutoProcessor.from_pretrained(model_id)

MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens):
    image = image.resize(MAX_IMAGE_SIZE)
    cleaned_output = ""

    prompt = f"<|image|><|begin_of_text|>{user_prompt} Answer:"
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    output = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
    )
    raw_output = processor.decode(output[0])
    cleaned_output = raw_output.replace("<|image|><|begin_of_text|>", "").strip().replace(" Answer:", "")

    return cleaned_output


def clear_chat():
    return ""


visual_theme = gr.themes.Default() 

def gradio_interface():
    with gr.Blocks() as demo:
        gr.HTML(
        """
        <h1 style='text-align: center; font-family: Arial, sans-serif;'>
        Demo LLama3.2-Vision
        </h1>
        <p style='text-align: center;'>Generate image descriptions</p>
        """)



        with gr.Row():
            # Cột trái với ảnh và các thanh trượt tham số
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=512,  
                    width=512   
                )

                # Các thanh trượt tham số
                temperature = gr.Slider(
                    label="Temperature", 
                    minimum=0.1, 
                    maximum=2.0, 
                    value=0.6, 
                    step=0.1, 
                    interactive=True
                )
                top_k = gr.Slider(
                    label="Top-k", 
                    minimum=1, 
                    maximum=100, 
                    value=50, 
                    step=1, 
                    interactive=True
                )
                top_p = gr.Slider(
                    label="Top-p", 
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.9, 
                    step=0.1, 
                    interactive=True
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", 
                    minimum=50, 
                    maximum=500, 
                    value=100, 
                    step=50, 
                    interactive=True
                )

            # Cột phải với giao diện chat
            with gr.Column(scale=2):
                
                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt here...", 
                    lines=2
                )

                # Các nút Generate và Clear
                with gr.Column():
                    with gr.Row():
                        generate_button = gr.Button("Generate")
                        clear_button = gr.Button("Clear")
                    
                    with gr.Row():
                        output_box = gr.Textbox(
                            label="Output",
                            show_label=True,
                            lines=10,
                            interactive=False,
                            placeholder="Your result will appear here...",
                            elem_classes="large-output"
                        )  

                    # Hành động của nút Generate
                    generate_button.click(
                        fn=describe_image, 
                        inputs=[image_input, user_prompt, temperature, top_k, top_p, max_tokens],
                        outputs=[output_box]
                    )
                    
                    # Hành động của nút Clear
                    clear_button.click(
                        fn=clear_chat,
                        inputs=[],
                        outputs=[output_box]
                    )
                
        # Đặt output_box ở dưới các nút
        return demo

# Launch the interface
demo = gradio_interface()
demo.launch()