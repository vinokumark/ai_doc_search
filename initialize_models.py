from doc_vectorizer import DocVectorizer

if __name__ == "__main__":
    print("Initializing DocVectorizer and downloading models...")
    try:
        vectorizer = DocVectorizer()
        print("Initialization complete. Models are downloaded and ready.")
    except Exception as e:
        print(f"An error occurred during initialization: {e}")
