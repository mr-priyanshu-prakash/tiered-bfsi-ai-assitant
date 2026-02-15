from app.pipeline import run_pipeline

def main():
    print("\n=== BFSI AI Assistant (Local Mode) ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User: ")

        if query.lower() == "exit":
            print("Exiting assistant.")
            break

        result = run_pipeline(query)

        print("\n--- Response ---")
        print("Tier:", result["tier"])
        print("Confidence:", result["confidence"])
        print("Assistant:", result["response"])
        print("----------------\n")

if __name__ == "__main__":
    main()
