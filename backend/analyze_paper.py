import os
from pypdf import PdfReader

PDF_PATH = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/02_python_eeg/2022 Li Deep Learning Sleep Stage Classification with EEG Spectrogram.pdf"

def analyze_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Error: File not found at {PDF_PATH}")
        return

    reader = PdfReader(PDF_PATH)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Search for keywords
    keywords = ["learning rate", "optimizer", "adam", "sgd", "decay", "momentum", "batch size", "epoch"]
    
    print(f"--- Analyzing {os.path.basename(PDF_PATH)} ---")
    
    # 1. Look for Method Section (heuristic)
    lower_text = text.lower()
    
    for kw in keywords:
        print(f"\n[Keyword: {kw}]")
        # Find all occurrences with context
        indices = [i for i in range(len(lower_text)) if lower_text.startswith(kw, i)]
        for idx in indices[:5]: # Limit to 5 matches per keyword
            start = max(0, idx - 100)
            end = min(len(text), idx + 100)
            print(f"...{text[start:end].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    analyze_pdf()
