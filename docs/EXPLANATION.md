Q1: Why did you choose this chunk size?
Answer: Answer: I implemented a chunking strategy with a size of 600 characters and a 100-character overlap.
Strategic Reasoning: A 600-character window (roughly 100 words) was chosen to balance semantic density and retrieval precision. It provides the LLM (Llama 3.3) with enough local context to understand the surrounding sentences without filling the context window with irrelevant information.
Overlap Necessity: I chose a 100-character overlap (~16%) to maintain contextual continuity. This ensures that if a key entity, such as a name or a specific value, falls at the boundary of a chunk, it is preserved in its entirety in the following chunk. This prevents "broken" sentences from degrading the quality of the vector embeddings.


Q2: One retrieval failure case you observed?
Answer: The system failed during Global Summarization.
Example: When I asked, "Summarize the entire 20-page document," the system only retrieved the top 5 most similar chunks. Because the AI didn't "see" the other 15 pages of data, the summary was only a summary of a few paragraphs, not the whole book. This is a common limitation of RAG.

Q3: One metric you tracked?
Answer: I tracked End-to-End Latency.
Observation: In the terminal logs, I noticed that Embedding & Retrieval (Pinecone) took ~200ms, while LLM Generation (Groq) took ~1.5 seconds. This confirmed that the bottleneck of the system is the AI's "thinking time," not the database search.