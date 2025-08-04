import os
import uuid
from typing import List, Dict, Optional
import PyPDF2
import docx
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from presidio_analyzer import AnalyzerEngine
from transformers import pipeline
import torch
import re
import os
import uuid
from typing import List, Dict, Optional
from unstructured.partition.auto import partition
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from presidio_analyzer import AnalyzerEngine
from sentence_transformers.cross_encoder import CrossEncoder

class DocVectorizer:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "documents"):
        self.client = Client(Settings(persist_directory=db_path, is_persistent=True))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        self.analyzer = AnalyzerEngine()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")

        # Generative and NER models for the RAG pipeline
        # We now load BOTH models to create a hybrid system.
        # The generative model provides fluent answers. Reverting to Gemma to test with new chunking.
        self.generative_pipe = pipeline("text-generation", model="google/gemma-2b-it", device=self.device, torch_dtype=torch.bfloat16)
        # The extractive model provides a reliable fallback.
        self.extractive_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=self.device)
        self.ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=self.device)

        # Using a re-ranking model specifically tuned for question-answering tasks.
        self.reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512, device=self.device)

    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language='en')
        redacted = list(text)
        for pii in results:
            for i in range(pii.start, pii.end):
                redacted[i] = "*"
        return "".join(redacted)

    def clear_database(self):
        """Clear all documents from the vector database."""
        try:
            # Delete the existing collection
            self.client.delete_collection(name=self.collection.name)
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_function
            )
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error clearing database: {e}")

    def add_document(self, file_path: str) -> list:
        """Processes a document using the unstructured library for robust semantic chunking."""
        from unstructured.chunking.by_title import chunk_by_title
        from unstructured.partition.auto import partition

        source_filename = os.path.basename(file_path)

        # Step 1: Use unstructured.io to partition the document into semantic elements.
        try:
            elements = partition(filename=file_path, strategy="auto")
        except Exception as e:
            raise ValueError(f"Failed to partition document: {e}")

        # Step 2: Use a semantic chunking strategy.
        try:
            raw_chunks = chunk_by_title(elements, max_characters=1024, combine_text_under_n_chars=256)
            chunks = [chunk.text.strip() for chunk in raw_chunks if chunk.text.strip()]
            
            # --- DEBUGGING: Show the created chunks --- #
            print(f"\n--- CHUNKING DEBUG ---")
            print(f"Successfully created {len(chunks)} chunks from {source_filename}:")
            for i, chunk in enumerate(chunks):
                print(f"  [Chunk {i+1}]: {chunk[:150].replace('\n', ' ')}...")
            print("--- END CHUNKING DEBUG ---\n")
            # --- END DEBUGGING --- #

            if not chunks:
                raise ValueError("No text chunks were extracted from the document.")
        except Exception as e:
            raise ValueError(f"Failed to chunk document: {e}")

        # Step 3: Prepare new data for ingestion.
        doc_ids = [str(uuid.uuid4()) for _ in chunks]
        documents = chunks
        metadatas = [{
            'source': source_filename,
            'category': 'text',
            'text_lowercase': chunk.lower()
        } for chunk in chunks]

        # Step 4: Atomically replace old chunks with new chunks in the database.
        try:
            # First, delete any existing chunks for this document.
            existing_ids = self.collection.get(where={"source": source_filename})['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)
            
            # Second, add the new chunks.
            self.collection.add(documents=documents, metadatas=metadatas, ids=doc_ids)
            print(f"Successfully processed and added {len(chunks)} chunks for {source_filename}.")
            return doc_ids
        except Exception as e:
            # This is a critical error, as the database might be in an inconsistent state.
            raise ConnectionError(f"Failed to update database for {source_filename}: {e}")

    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language='en')
        redacted = list(text)
        for pii in results:
            for i in range(pii.start, pii.end):
                redacted[i] = "*"
        return "".join(redacted)



    def search(self, query: str, n_results: int = 5, redact_pii: bool = True) -> List[Dict]:
        """Performs a hybrid search using NER and a QA model to extract a factual answer."""
        print(f"DEBUG: Query: {query}")

        # Step 1: Use NER to detect persons in the query and clean the names
        detected_persons = []
        try:
            entities = self.ner_pipe(query)
            for entity in entities:
                if entity['entity_group'] == 'PER' and len(entity['word']) > 2:
                    # Clean the word: remove hashtags and possessive 's
                    clean_word = entity['word'].replace('#', '').strip()
                    if clean_word.endswith("'s"):
                        clean_word = clean_word[:-2]
                    
                    if clean_word:
                        detected_persons.append(clean_word.lower())
            print(f"DEBUG: NER detected persons: {detected_persons}")
        except Exception as e:
            print(f"DEBUG: NER failed: {e}")

        # Step 2: Perform a targeted or general search
        # Step 2: Perform a broad semantic search first.
        # We will filter the results in code to bypass database limitations.
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 10,  # Get a larger pool of candidates
        )

        # Step 3: Manually filter the results for keywords for robustness.
        stop_words = {'what', 'is', 'the', 'of', 'a', 'an', 'in', 'for', 'to', 'show', 'me', 'tell'}
        query_words = [word.lower().replace("'s", "") for word in query.split() if word.lower() not in stop_words and len(word) > 2]
        
        filtered_results = []
        if query_words:
            for i, doc in enumerate(results['documents'][0]):
                doc_lower = doc.lower()
                # Check if any keyword is in the document.
                if any(keyword in doc_lower for keyword in query_words):
                    # Reconstruct the result entry for the filtered list
                    filtered_results.append({
                        'document': doc,
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            print(f"DEBUG: Found {len(filtered_results)} documents after manual filtering.")

        # If filtering produced results, use them. Otherwise, use the original results.
        if filtered_results:
            final_documents = [res['document'] for res in filtered_results]
            final_ids = [res['id'] for res in filtered_results]
            final_metadatas = [res['metadata'] for res in filtered_results]
        else:
            final_documents = results['documents'][0]
            final_ids = results['ids'][0]
            final_metadatas = results['metadatas'][0]

        # Replace the original results with the filtered ones for the next step.
        results = {
            'documents': [final_documents],
            'ids': [final_ids],
            'metadatas': [final_metadatas]
        }

        if not results or not results['documents'] or not results['documents'][0]:
            return [{'id': 'none', 'score': 0, 'document': 'No relevant information found.', 'metadata': {}}]

        # Step 2: Synthesize context and extract the answer
        # This is where the magic happens. We combine the chunks and use the QA model.
        final_answer, best_doc_id, best_metadata, best_doc = self._synthesize_and_answer(
            query=query,
            documents=results['documents'][0],
            ids=results['ids'][0],
            metadatas=results['metadatas'][0]
        )

        # Step 3: Apply PII redaction if requested
        if redact_pii:
            final_answer = self.redact_pii(final_answer)

        return [{
            'id': best_doc_id,
            'score': 1.0,  # Score is less relevant now as we have a factual answer
            'document': final_answer,
            'metadata': best_metadata,
            'context': best_doc # Return the context for debugging
        }]

    def _synthesize_and_answer(self, query: str, documents: List[str], ids: List[str], metadatas: List[Dict]) -> (str, str, Dict, str):
        """Implements the Hybrid Model Cascade for robust, high-quality answers."""

        # Step 1: Re-rank documents to find the best context.
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, documents, ids, metadatas), reverse=True)
        best_doc, best_id, best_metadata = scored_docs[0][1], scored_docs[0][2], scored_docs[0][3]
        print(f"DEBUG: Top re-ranked document: {best_doc[:300]}...")

        # Step 2: Attempt to get a generative answer from Gemma.
        try:
            # Using Gemma's prompt format with strict instructions to prevent hallucination.
            prompt = f"<start_of_turn>user\nStrictly answer the question using only the provided context. Do not use any other knowledge.\n\nContext:\n{best_doc}\n\nQuestion:\n{query}<end_of_turn>\n<start_of_turn>model\n"
            outputs = self.generative_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.01)
            # The answer from Gemma is the text that comes after the final model turn token.
            answer = outputs[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()
            print(f"DEBUG: Gemma model answer: {answer}")

            # Check for a cautious refusal.
            refusal_phrases = ["cannot answer", "not mentioned", "don't know", "does not contain"]
            if not any(phrase in answer.lower() for phrase in refusal_phrases):
                return answer, best_id, best_metadata, best_doc
            
            print("DEBUG: Gemma refused to answer. Falling back to extractive model.")

        except Exception as e:
            print(f"Error during generative pipeline: {e}. Falling back.")

        # Step 3: Fallback to the reliable extractive model (Distilbert).
        try:
            qa_input = {'question': query, 'context': best_doc}
            result = self.extractive_pipe(qa_input)
            answer = result['answer']
            print(f"DEBUG: Extractive model answer: {answer}, score: {result['score']}")

            if result['score'] < 0.1:
                return "The document may contain an answer, but I am not confident.", best_id, best_metadata, best_doc
            
            return answer, best_id, best_metadata, best_doc

        except Exception as e:
            print(f"Error during extractive pipeline: {e}")
            return "Could not extract an answer from the document.", best_id, best_metadata, best_doc

        for offset in [-adjacent_range, adjacent_range]:
            check_idx = line_index + offset
            if 0 <= check_idx < len(all_lines):
                adjacent_line = all_lines[check_idx]
                if person_name in adjacent_line.lower():
                    return True
        
        # 3. Content density - lines with substantial information
        # Prefer lines that have meaningful content (not just headers or whitespace)
        if len(line.strip()) > 20 and any(char.isdigit() for char in line):
            return True
        
        # 4. Document structure patterns - common across all enterprise docs
        # Lines that start with common prefixes in structured documents
        line_stripped = line.strip()
        if line_stripped and (line_stripped[0].isupper() or line_stripped.startswith(('-', '•', '*'))):
            return True
        
        # 5. Semantic indicators - words that suggest important information
        # Use general semantic indicators that work across document types
        semantic_indicators = [
            # General relationship indicators
            'is', 'was', 'has', 'have', 'will', 'can', 'should',
            # Contact/identity indicators  
            'contact', 'id', 'number', 'address',
            # Status/attribute indicators
            'status', 'type', 'level', 'grade', 'class',
            # Action/responsibility indicators
            'responsible', 'assigned', 'manages', 'leads', 'works',
            # Time/date indicators
            'date', 'time', 'when', 'since', 'until'
        ]
        
        # Check if line contains semantic indicators
        if any(indicator in line_lower for indicator in semantic_indicators):
            return True
        
        # Default: not contextually related
        return False
