# trim_pdfs.py
from pypdf import PdfWriter, PdfReader

def trim_pdf(path, max_pages=15):
    reader = PdfReader(path)
    original_count = len(reader.pages)
    writer = PdfWriter()
    for page in reader.pages[:max_pages]:
        writer.add_page(page)
    with open(path, "wb") as f:
        writer.write(f)
    print(f"{path}: {original_count} -> {min(max_pages, original_count)} pages")

trim_pdf("scripts/sample_corpus/pdf1/github_actions.pdf", max_pages=15)
trim_pdf("scripts/sample_corpus/pdf1/wellarchitected-aws.pdf", max_pages=15)