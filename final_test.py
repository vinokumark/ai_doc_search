from doc_vectorizer import DocVectorizer

def run_tests():
    """Runs the final user-provided test cases against the live system."""

    print("--- Starting Final System Test ---")
    vectorizer = DocVectorizer()

    test_queries = [
        "what is role vinoth",
        "show full information janani customer",
        "show only email id of vinoth",
        "what is role of sweetha in bts"
    ]

    for query in test_queries:
        print(f"\n--- Testing Query: '{query}' ---")
        try:
            results = vectorizer.search(query, redact_pii=False)
            if results and results[0]['document']:
                print("\n[RESULT]")
                print(results[0]['document'])
                print("\n")
            else:
                print("\n[RESULT] No document found.\n")
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}\n")

    print("--- Test Complete ---")

if __name__ == "__main__":
    run_tests()
