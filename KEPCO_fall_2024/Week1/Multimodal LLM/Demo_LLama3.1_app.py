from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import gradio as gr


max_seq_length = 2048 
dtype = None 
load_in_4bit = False


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir="/workspace/nmquy/hf_cache"
)

FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


def generate_stream(instruction_prompt, input_prompt, temperature, top_k, top_p, max_tokens):
    # Chuẩn bị dữ liệu đầu vào cho mô hình
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                f"{instruction_prompt}",  # instruction
                f"{input_prompt}",  # input
                "",  # output - để trống cho quá trình sinh văn bản
            )
        ], return_tensors="pt").to("cuda")
    
    # Sử dụng TextStreamer để xử lý việc streaming
    text_streamer = TextStreamer(tokenizer)
    
    # Sinh văn bản và sử dụng streamer để lấy kết quả từng phần
    generated_output = model.generate(
        **inputs, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        streamer=text_streamer, 
        max_new_tokens=max_tokens
    )
    raw_output = tokenizer.decode(generated_output[0])
    cleaned_output = raw_output.replace("<|image|><|begin_of_text|>", "").strip().replace(" Answer:", "")
    response_part = cleaned_output.split("### Response:")[1].strip()
    return response_part

def clear_chat():
    return ""

visual_theme = gr.themes.Default() 


def gradio_interface():
    with gr.Blocks() as demo:
        gr.HTML(
        """
        <h1 style='text-align: center; font-family: Arial, sans-serif;'>
        Demo LLaMA3.1-8B 
        </h1>
        <p style='text-align: center;'>Easily generate text with LLaMA</p>
        """)
        with gr.Row():
            with gr.Column(scale=1):
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
            with gr.Column(scale=2):
                
                instruction_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your instruction prompt here...", 
                    lines=2
                )
                
                input_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your input prompt here...", 
                    lines=2
                )
                
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
                    generate_button.click(
                        fn=generate_stream, 
                        inputs=[instruction_prompt, input_prompt, temperature, top_k, top_p, max_tokens],
                        outputs=[output_box]
                    )
                    
                    clear_button.click(
                        fn=clear_chat,
                        inputs=[],
                        outputs=[output_box]
                    )
        return demo

demo = gradio_interface()
demo.launch()
