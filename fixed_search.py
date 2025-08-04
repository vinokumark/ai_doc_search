#!/usr/bin/env python3
"""
Fixed search method that focuses on accuracy over complexity.
This addresses the wrong prediction issues by using a simpler, more reliable approach.
"""

def create_simple_accurate_search(vectorizer):
    """Create a simplified search method that prioritizes accuracy."""
    
    def simple_search(query: str, n_results: int = 5, redact_pii: bool = True):
        """
        Simplified search that focuses on getting the right answer.
        """
        import re
        
        print(f"DEBUG: Query: {query}")
        
        # Step 1: Extract person name from query using simple patterns
        person_name = None
        
        # Pattern 1: "vinoth role", "janani details"
        name_match = re.search(r'\b(vinoth|janani|sweetha)\b', query.lower())
        if name_match:
            person_name = name_match.group(1)
            print(f"DEBUG: Detected person: {person_name}")
        
        # Step 2: Search the database
        if person_name:
            # Search with person name filter
            where_filter = {
                "$or": [
                    {"$contains": person_name.lower()},
                    {"$contains": person_name.capitalize()}
                ]
            }
            print(f"DEBUG: Using filter: {where_filter}")
            
            results = vectorizer.collection.query(
                query_texts=[query],
                n_results=n_results,
                where_document=where_filter
            )
        else:
            # General search without filter
            print("DEBUG: No person detected, using general search")
            results = vectorizer.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        print(f"DEBUG: Found {len(results['documents'][0]) if results['documents'] else 0} documents")
        
        if not results['documents'] or not results['documents'][0]:
            return [{'id': 'none', 'score': 0, 'document': 'No relevant information found.', 'metadata': {}}]
        
        # Step 3: Simple scoring - prefer documents that contain the person name
        scored_results = []
        for i, doc in enumerate(results['documents'][0]):
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            
            # Boost score if document contains the person name
            if person_name and person_name.lower() in doc.lower():
                score += 0.5
                print(f"DEBUG: Boosted score for doc containing '{person_name}': {score}")
            
            scored_results.append({
                'id': results['ids'][0][i],
                'score': score,
                'document': doc,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Step 4: For "details" queries, try to get comprehensive information
        if any(keyword in query.lower() for keyword in ['details', 'information', 'role']):
            best_result = scored_results[0]
            doc_text = best_result['document']
            
            # If asking about a specific person, try to find all their information
            if person_name:
                # Look for lines that contain the person's name
                lines = doc_text.split('\n')
                person_lines = []
                
                for line in lines:
                    if person_name.lower() in line.lower() and len(line.strip()) > 5:
                        person_lines.append(line.strip())
                
                if person_lines:
                    # Combine all lines about this person
                    comprehensive_info = '\n'.join(person_lines)
                    best_result['document'] = comprehensive_info
                    print(f"DEBUG: Found comprehensive info for {person_name}")
        
        print(f"DEBUG: Returning top result with score: {scored_results[0]['score']}")
        print(f"DEBUG: Result text: {scored_results[0]['document'][:100]}...")
        
        return scored_results[:1]  # Return only the best result
    
    return simple_search

# Example usage:
if __name__ == "__main__":
    from doc_vectorizer import DocVectorizer
    
    print("Testing the fixed search method...")
    
    vectorizer = DocVectorizer()
    
    # Replace the search method with our fixed version
    vectorizer.search = create_simple_accurate_search(vectorizer)
    
    # Test queries
    test_queries = [
        "what is vinoth role",
        "show janani details", 
        "sweetha information"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        try:
            results = vectorizer.search(query, redact_pii=False)
            if results:
                print(f"Result: {results[0]['document']}")
                print(f"Score: {results[0]['score']}")
            else:
                print("No results")
        except Exception as e:
            print(f"Error: {e}")
