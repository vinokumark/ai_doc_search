#!/usr/bin/env python3
"""
Simple test script to debug and fix the Document AI search issues.
This will help identify why the system is returning wrong predictions.
"""

from doc_vectorizer import DocVectorizer
import json

def test_search_accuracy():
    """Test the search system with known queries and expected results."""
    
    print("=== Document AI Search Accuracy Test ===\n")
    
    # Initialize the vectorizer
    vectorizer = DocVectorizer()
    
    # Test cases with expected results
    test_cases = [
        {
            "query": "what is vinoth role",
            "expected_keywords": ["GCP admin", "vinoth"],
            "should_not_contain": ["Director", "BTS", "sweetha"]
        },
        {
            "query": "show janani details",
            "expected_keywords": ["janani", "tsdf@gmail.com", "123456343"],
            "should_not_contain": ["vinoth", "sweetha"]
        },
        {
            "query": "sweetha role",
            "expected_keywords": ["sweetha", "Director", "BTS Fan"],
            "should_not_contain": ["janani", "vinoth", "GCP"]
        }
    ]
    
    print("Testing search accuracy...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test_case['query']} ---")
        
        try:
            # Perform the search
            results = vectorizer.search(test_case['query'], n_results=3, redact_pii=False)
            
            if results and len(results) > 0:
                result_text = results[0]['document'].lower()
                print(f"Result: {results[0]['document'][:200]}...")
                print(f"Score: {results[0]['score']}")
                
                # Check if expected keywords are present
                expected_found = []
                for keyword in test_case['expected_keywords']:
                    if keyword.lower() in result_text:
                        expected_found.append(keyword)
                
                # Check if unwanted keywords are present
                unwanted_found = []
                for keyword in test_case['should_not_contain']:
                    if keyword.lower() in result_text:
                        unwanted_found.append(keyword)
                
                # Evaluate the result
                if len(expected_found) >= 1 and len(unwanted_found) == 0:
                    print("✅ PASS: Found expected content, no unwanted content")
                elif len(expected_found) >= 1 and len(unwanted_found) > 0:
                    print(f"⚠️  PARTIAL: Found expected {expected_found} but also unwanted {unwanted_found}")
                else:
                    print(f"❌ FAIL: Missing expected keywords or found unwanted content")
                    print(f"   Expected: {test_case['expected_keywords']}")
                    print(f"   Found expected: {expected_found}")
                    print(f"   Found unwanted: {unwanted_found}")
                
            else:
                print("❌ FAIL: No results returned")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
        
        print()
    
    print("=== Test Complete ===")
    print("\nRecommendations based on test results:")
    print("1. If tests are failing, the vector search or re-ranking is not working correctly")
    print("2. Check if the document was uploaded correctly")
    print("3. Verify that entity detection is finding the right names")
    print("4. Consider simplifying the search to focus on exact text matching")

def simple_database_inspection():
    """Inspect what's actually in the database."""
    
    print("\n=== Database Inspection ===")
    
    vectorizer = DocVectorizer()
    
    try:
        # Get all documents in the collection
        all_docs = vectorizer.collection.get()
        
        print(f"Total documents in database: {len(all_docs['documents']) if all_docs['documents'] else 0}")
        
        if all_docs['documents']:
            print("\nFirst 5 documents in database:")
            for i, doc in enumerate(all_docs['documents'][:5]):
                print(f"Doc {i+1}: {doc[:100]}...")
                if all_docs['metadatas'] and len(all_docs['metadatas']) > i:
                    print(f"   Metadata: {all_docs['metadatas'][i]}")
                print()
        
    except Exception as e:
        print(f"Error inspecting database: {e}")

if __name__ == "__main__":
    # Run the tests
    test_search_accuracy()
    simple_database_inspection()
