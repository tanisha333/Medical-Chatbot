import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS"
    / "research"
    / "trials.ipynb"
)
OUTPUT = ROOT / "data" / "medical_book_docs.jsonl"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text_output = "".join(
        "".join(output.get("data", {}).get("text/plain", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )

    pattern = re.compile(
        r"Document\(metadata=(\{.*?\}), page_content="
        r"((?:'(?:[^'\\]|\\.)*')|(?:\"(?:[^\"\\]|\\.)*\"))\)",
        re.DOTALL,
    )

    rows = []
    seen_pages = set()
    for match in pattern.finditer(text_output):
        metadata = ast.literal_eval(match.group(1))
        page_content = ast.literal_eval(match.group(2))
        page = metadata.get("page")
        if not isinstance(page, int):
            continue
        if page in seen_pages:
            continue
        seen_pages.add(page)
        rows.append(
            {
                "page_content": page_content,
                "metadata": {
                    "source": "data\\Medical_book.pdf",
                    "page": page,
                    "page_label": metadata.get("page_label"),
                },
            }
        )

    if not rows:
        raise RuntimeError(f"No document rows found in {NOTEBOOK}")

    OUTPUT.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    nonempty = sum(bool(row["page_content"].strip()) for row in rows)
    print(f"Wrote {len(rows)} docs ({nonempty} non-empty) to {OUTPUT}")


if __name__ == "__main__":
    main()
