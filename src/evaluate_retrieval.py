from src.search import model, collection

TEST_CASES = [
    {
        "question": "What information should customers provide when requesting a replacement for a defective product?",
        "expected_source": "product_policy.txt",
        "expected_section": "Returns and Replacements",
    },
    {
        "question": "What types of damage may not be covered by a manufacturer's warranty?",
        "expected_source": "product_policy.txt",
        "expected_section": "Product Warranty and Defective Items",
    },
    {
        "question": "How long does an approved refund normally take to process?",
        "expected_source": "customer_support.txt",
        "expected_section": "Refunds",
    },
    {
        "question": "How many working days of annual leave do full-time employees receive?",
        "expected_source": "hr_policy.txt",
        "expected_section": "Annual Leave",
    },
    {
        "question": "Does artificial intelligence replace the judgement of qualified medical professionals?",
        "expected_source": "ai_healthcare.txt",
        "expected_section": None,
    },
]


def evaluate_retrieval(k=2):
    passed = 0

    for test in TEST_CASES:
        embedding = model.encode(
            [test["question"]],
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=embedding,
            n_results=min(k, collection.count()),
            include=["metadatas", "documents"]
        )

        retrieved = results["metadatas"][0]

        found = any(
            item["source"] == test["expected_source"]
            and (
                test["expected_section"] is None
                or item["section"] == test["expected_section"]
            )
            for item in retrieved
        )

        if found:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{status}] {test['question']}")
        print(f"Expected: {test['expected_source']} | {test['expected_section']}")
        print("Retrieved:")

        for item in retrieved:
            print(f"  - {item['source']} | {item['section']}")

    total = len(TEST_CASES)
    score = (passed / total) * 100

    print("\n" + "=" * 45)
    print(f"Retrieval Recall@{k}: {passed}/{total} ({score:.1f}%)")
    print("=" * 45)


if __name__ == "__main__":
    evaluate_retrieval()
