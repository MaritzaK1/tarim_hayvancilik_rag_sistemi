import fitz
def extract_text_from_pdf(file_path: str) -> dict:
    doc = fitz.open(file_path)
    result = {
        "filename": file_path.split("\\")[-1].split("/")[-1],
        "total_pages": len(doc),
        "pages": []
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            result["pages"].append({
                "page_num": page_num + 1,
                "text": text.strip()
            })

    doc.close()
    return result
