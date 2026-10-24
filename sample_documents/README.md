# Sample Documents

This folder contains sample documents you can use to test and demonstrate the Enterprise RAG System.

## Files

| File | Description |
|------|-------------|
| `technova_handbook.md` | TechNova Solutions Employee Handbook (Markdown format) |

## How to Use

1. **Convert to PDF** – The RAG system requires a PDF file.
   - Open `technova_handbook.md` in any Markdown viewer or paste its content into [Word](https://word.com) / Google Docs, then export as PDF.
   - Alternatively use a free online converter such as [md2pdf](https://md2pdf.netlify.app/) or [Pandoc](https://pandoc.org).
   - On Windows with Pandoc: `pandoc technova_handbook.md -o technova_handbook.pdf`

2. **Upload** the generated PDF in the Streamlit app.

3. **Ask demo questions** (see README.md for the full list):

   - *"How many paid annual leave days are employees entitled to?"*
   - *"What are the company's working hours?"*
   - *"Can employees work from home?"*
   - *"What is the notice period?"*
   - *"Who is covered under the health insurance?"*
   - *"What is the probation period?"*

4. **Test hallucination prevention:**
   - *"Who is the President of the United States?"*
   - Expected: "I couldn't find this information in the uploaded document."
