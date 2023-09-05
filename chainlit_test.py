from langchain import PromptTemplate, LLMChain, LlamaCpp
import chainlit as cl

number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ###Instruction:
    You are an expert witness specializing in empathy, toxicity, and professionalism.
    Given a person's message history and a current message as context, rate the messages on a scale of 1-100 for how professional they are (higher scores indicate more professional messages).
    Please respond with only an integer between 1 and 100, then give a short explanation of how the person could be more professional.

    ###Input:
    Message History: "test"

    Current Message:
    {context}


    ###Response:
    Your Professionalism rating from 1-100 is """

test_template = """
Test:
{context}
"""

llm = LlamaCpp(
            model_path="./models/losslessmegacoder-llama2-13b-q2_k.bin",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=8192,
            verbose=True
            )

@cl.on_chat_start
def main():
    #Instantiate the chain
    prompt = PromptTemplate(template=number_template, input_variables=["context"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain") #type: LLMChain

    # Call the chain asyncronously
    res = await cl.make_async(llm_chain)(message, callbacks=[cl.LangchainCallbackHandler()])

    # Do post processing and RAG

    # 'res' is a Dict
    await cl.Message(content=res['text']).send()
    return llm_chain