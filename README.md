# PerspecToolformer

## Quickstart installation:
```conda env create --name envname --file=environments.yml```

## Start the testing app using:
```gradio test_app.py```

## Contribution:
- Better indexing and distance functions for RAG
- Asynchronous calls to LLM to give users the immediate insight of similarly rated messages and LLM insight later (likely on next login)
- NTK aware rope scaling up to 16k tokens without significant perplexity jump
-  More labeled examples to index in toxicity.csv and professionalism.txt in the data folder
- Faster and more efficient slack api integration
- More API message sources, e.g. Reddit, Teams, Discord, etc.
