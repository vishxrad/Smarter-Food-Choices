from typing import Callable
import gradio as gr
from PIL import Image
import pytesseract  

# Example messages for the QA chatbot
english_examples = [
    ["Does this contain any preservatives?"],
    ["How can I make my salad high in protein without adding meat?"],
    ["Is this food item healthy?"],
    ["What are some gluten-free options for pasta?"],
    ["What can I eat with grilled chicken to make it a balanced dinner?"]
]

def clear_files():
    return "Vector Store is Not ready"

def handle_user_message(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def extract_text_from_image(image):
    """Perform OCR on the uploaded image and return extracted text."""
    text = pytesseract.image_to_string(image)
    return text

def make_demo(
    load_doc_fn: Callable,
    run_fn: Callable,
    stop_fn: Callable,
    update_retriever_fn: Callable,
    model_name: str,
):
    examples = english_examples
    text_example_path = "text_example_en.pdf"

    with gr.Blocks(
        theme='Nymbo/Nymbo_Theme',
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown("""<h1><center>Smarter Food Choices with RAG and OCR</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO and {model_name} </center>""")

        with gr.Row():
            with gr.Column(scale=1):
                docs = gr.File(
                    label="Step 1: Load text files",
                    value=[text_example_path],
                    file_count="multiple",
                    file_types=[
                        ".csv", ".doc", ".docx", ".enex", ".epub",
                        ".html", ".md", ".odt", ".pdf", ".ppt",
                        ".pptx", ".txt",
                    ],
                )
                image_input = gr.Image(
                    label="Step 1: Upload an image for OCR",
                    type="pil",
                )
                
                load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
                db_argument = gr.Accordion("Vector Store Configuration", open=False)

                with db_argument:
                    spliter = gr.Dropdown(
                        ["Character", "RecursiveCharacter", "Markdown"],
                        value="RecursiveCharacter",
                        label="Text Spliter",
                        info="Method used to split the documents",
                        multiselect=False,
                    )

                    chunk_size = gr.Slider(
                        label="Chunk size",
                        value=400,
                        minimum=50,
                        maximum=2000,
                        step=50,
                        interactive=True,
                        info="Size of sentence chunk",
                    )

                    chunk_overlap = gr.Slider(
                        label="Chunk overlap",
                        value=50,
                        minimum=0,
                        maximum=400,
                        step=10,
                        interactive=True,
                        info="Overlap between 2 chunks",
                    )

                langchain_status = gr.Textbox(
                    label="Vector Store Status",
                    value="Vector Store is Ready",
                    interactive=False,
                )
                do_rag = gr.Checkbox(
                    value=True,
                    label="RAG is ON",
                    interactive=True,
                    info="Whether to do RAG for generation",
                )

            with gr.Accordion("Generation Configuration", open=False):
                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.1,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                    with gr.Column():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=1.0,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                    with gr.Column():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=50,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                    with gr.Column():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.1,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
        
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=800,
                    label="Step 3: Input Query",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="QA Message Box",
                        placeholder="Chat Message Box",
                        show_label=False,
                        container=False,
                    )
                    submit = gr.Button("Submit", variant="primary")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")

                gr.Examples(examples, inputs=msg, label="Click on any example and press 'Submit'")

                extract_text_button = gr.Button("Extract Text from Image")
                extract_text_button.click(fn=extract_text_from_image, inputs=image_input, outputs=msg)

                docs.clear(clear_files, outputs=[langchain_status], queue=False)
                load_docs.click(
                    fn=load_doc_fn,
                    inputs=[docs, spliter, chunk_size, chunk_overlap],
                    outputs=[langchain_status],
                    queue=False,
                )

                submit_event = msg.submit(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                    run_fn,
                    [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
                    chatbot,
                    queue=True,
                )
                
                submit_click_event = submit.click(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                    run_fn,
                    [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
                    chatbot,
                    queue=True,
                )

                stop.click(fn=stop_fn, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False)
                clear.click(lambda: None, None, chatbot, queue=False)

    return demo
