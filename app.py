from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain, SimpleSequentialChain

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import  ConversationSummaryBufferMemory, ConversationBufferWindowMemory

import gradio as gr
import json


toxic_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction:
You are an expert psychologist with a specialty in sociology.
Given someone's private message history, you can tell them how to improve their empathy and decrease their toxicity citing examples from the text.
You are constructively critical and will avoid being too nice, as that could increase toxicity. Professionalism dictates that nearly everyone can improve.
If you are unable to find anything, you will truthfully tell them they are doing great or that you don't know.

###Input:
Message History: {summary}
{new_lines}

###Response:
Psychologist: Here's how you can improve:"""

toxic_prompt_template = PromptTemplate(input_variables=["summary", "new_lines"], template=toxic_template)

llm = LlamaCpp(
    model_path="./models/hermes-llongma-2-7b-8k.ggmlv3.q2_k.bin",
    n_gpu_layers=0,
    n_batch=512,
    n_ctx=8000,
    verbose=True
    )

toxicity_chain = LLMChain(llm=llm, prompt=toxic_prompt_template)

number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction:
You are an expert witness specializing in toxicity and empathy.
Given a summary from another expert, rate the messages on a scale of 1-100 for how toxic they are.
Please respond with only an integer between 1 and 100.

###Input:


###Response:
Review from a New York Times play critic of the above play:"""

number_prompt_template = PromptTemplate(input_variables=["synopsis"], template=number_template)

review_chain = LLMChain(llm=llm, prompt=number_prompt_template)

# This is the overall chain where we run these two chains in sequence.

overall_chain = SimpleSequentialChain(chains= [toxicity_chain, review_chain],
                                      verbose=True)





def predict_convo(input=""):
    review = overall_chain.run(input)
    return json.dump({"review": review})

with gr.Blocks() as app:
    messages = gr.Textbox(label="messages")

    gr.Examples(
        [
            ["You're the best!\nWe can do it.\nYou've never been good enough.\nI can take care of that whenever.\nsup bro\nHow do we know when we're done?\nAre you going to be in the office today?\nHow long does it take to get there?\nI'll be there in about 30 minutes\nWhy are you always constantly late?"],
        ],
        inputs=[messages]
    )

    start_btn = gr.Button("Detect Toxicity")
    start_btn.click(
        fn=predict_convo,
        inputs=[messages],
        outputs=gr.Textbox(label="output"),
        api_name="toxicitydetect"
    )
    
app.queue(concurrency_count=1)
app.launch()#inbrowser=True, share=True,