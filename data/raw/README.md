# BABOK PDF Placeholder

## 📄 Place Your BABOK PDF Here

This folder is where you should place your BABOK guide PDF file.

### Instructions:

1. **Locate your BABOK PDF** on your computer
2. **Copy it to this folder** (`data/raw/`)
3. **Recommended filename**: `BABOK_v3.pdf` (or whatever version you have)

### Supported Formats:
- ✅ PDF files (`.pdf`)
- ❌ Word documents (`.docx`) - convert to PDF first
- ❌ ePub or other formats - convert to PDF first

### After Adding the PDF:

Run the data loader test to verify it works:

```bash
python src/data_loader.py
```

This will:
- Detect your BABOK PDF
- Extract text from all pages
- Show you a sample of the extracted content

---

**Note:** The BABOK guide is copyrighted material. Make sure you have a legal copy before using this system.
