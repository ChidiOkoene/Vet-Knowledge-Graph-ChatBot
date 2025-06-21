# pdf_preprocessor.py

# Ensure Tesseract is installed and pytesseract can find it
# For Unix: sudo apt-get install tesseract-ocr
# For Windows: download installer and update PATH


class PDFPreprocessor:
    def __init__(self, input_dir, output_dir, ocr_threshold=10):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ocr_threshold = ocr_threshold  # min text per page to skip OCR
        os.makedirs(output_dir, exist_ok=True)

    def extract_with_pdfplumber(self, path):
        """
        Extract text page-by-page using pdfplumber.
        """
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                texts.append(text)
        return texts

    def extract_images_with_pymupdf(self, doc, page_number, zoom=2):
        """
        Render page to image for OCR.
        """
        page = doc.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def ocr_page(self, img: Image.Image):
        """
        Perform OCR on PIL Image.
        """
        return pytesseract.image_to_string(img)

    def clean_text(self, text):
        """
        Remove headers/footers and normalize whitespace.
        """
        # Example: remove lines shorter than 3 characters or page numbers
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            # strip page numbers
            if re.match(r"^\s*\d+\s*$", line):
                continue
            # remove very short lines
            if len(line.strip()) < 3:
                continue
            cleaned.append(line.strip())
        return "\n".join(cleaned)

    def segment_into_chunks(self, text, max_chars=1000):
        """
        Split text into overlapping chunks ~max_chars length.
        """
        sentences = sent_tokenize(text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current += ' ' + sent
            else:
                chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    def process_pdf(self, filename):
        path = os.path.join(self.input_dir, filename)
        base, _ = os.path.splitext(filename)
        texts = self.extract_with_pdfplumber(path)

        # Open with PyMuPDF once for OCR
        doc = fitz.open(path)

        all_chunks = []
        for i, page_text in enumerate(texts):
            if len(page_text.strip()) < self.ocr_threshold:
                # likely scanned, perform OCR
                img = self.extract_images_with_pymupdf(doc, i)
                page_text = self.ocr_page(img)
            clean = self.clean_text(page_text)
            chunks = self.segment_into_chunks(clean)

            # Save per-page chunks
            for j, chunk in enumerate(chunks):
                out_file = f"{base}_page{i+1}_chunk{j+1}.txt"
                with open(os.path.join(self.output_dir, out_file), 'w', encoding='utf-8') as f:
                    f.write(chunk)
                all_chunks.append(out_file)
        doc.close()
        return all_chunks

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        summary = {}
        for pdf_file in files:
            chunks = self.process_pdf(pdf_file)
            summary[pdf_file] = chunks
        return summary



