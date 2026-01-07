import os

DASHBOARD_PATH = "frontend/src/components/ProjectViewer/MatlabDashboard.jsx"
TRAIN_PATH = "02_python_eeg/train.py"
MODELS_PATH = "02_python_eeg/models.py"
DATA_LOADER_PATH = "02_python_eeg/data_loader.py"

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def update_dashboard():
    dashboard = read_file(DASHBOARD_PATH)
    train_code = read_file(TRAIN_PATH)
    models_code = read_file(MODELS_PATH)
    data_loader_code = read_file(DATA_LOADER_PATH)
    
    # 1. Update train.py
    # Marker: <span className="font-mono text-xs text-gray-400">train.py (PyTorch/MPS)</span>
    # Followed by <SyntaxHighlighter ...> {`
    
    new_dashboard = replace_code_block(dashboard, "train.py (PyTorch/MPS)", train_code, "python")
    new_dashboard = replace_code_block(new_dashboard, "models.py (EEGSNet Architecture)", models_code, "python")
    new_dashboard = replace_code_block(new_dashboard, "data_loader.py (HDF5 Handling)", data_loader_code, "python")
    
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(new_dashboard)
    print("Dashboard updated successfully.")

def replace_code_block(content, marker, new_code, lang):
    # Find marker
    marker_idx = content.find(marker)
    if marker_idx == -1:
        print(f"Marker not found: {marker}")
        return content
        
    # Find start of Template Literal inside SyntaxHighlighter after marker
    # Look for `{`
    start_tag = "<SyntaxHighlighter"
    start_tag_idx = content.find(start_tag, marker_idx)
    
    # Find `{` after start_tag
    open_brace_idx = content.find("{`", start_tag_idx)
    if open_brace_idx == -1:
         # Maybe spaced { `
         open_brace_idx = content.find("{ `", start_tag_idx)
         
    if open_brace_idx == -1:
        print(f"Opening brace not found after {marker}")
        return content
        
    # Find closing `}`
    # We need to find `}` that closes the template literal.
    # It corresponds to lines ending with `}` before `</SyntaxHighlighter>`
    # But since code may contain `}`, we look for `}` just before `</SyntaxHighlighter>`
    
    close_tag = "</SyntaxHighlighter>"
    close_tag_idx = content.find(close_tag, open_brace_idx)
    
    # Search backwards from close_tag for `}`
    close_brace_idx = content.rfind("`}", open_brace_idx, close_tag_idx)
    if close_brace_idx == -1:
         # Try just `}`
         close_brace_idx = content.rfind("}", open_brace_idx, close_tag_idx)
    
    if close_brace_idx == -1:
         print(f"Closing brace not found for {marker}")
         return content

    # Construct new block
    # We preserve: content[:open_brace_idx] + "{`" + new_code + "`}" + content[close_brace_idx+2:]
    # Wait, check if existing used `}` or `}` ?
    # From file view, it uses `{` ... `}`
    
    pre = content[:open_brace_idx]
    post = content[close_tag_idx:] # From </SyntaxHighlighter> onwards
    
    # We need to close the brace properly.
    # The original was `{`CODE`}` (backticks + brace)
    # So we write `{` + `new_code` + `}` + match spacing
    # Actually, simpler: replace everything between `open_brace_idx` and `close_tag_idx`
    
    # Determine indentation of `</SyntaxHighlighter>` to align `}`
    # Usually `                                    </SyntaxHighlighter>` (36 spaces)
    indent = "                                    " # Estimate
    
    # Escape backticks and ${ to prevents JSX template literal breakage
    new_code = new_code.replace("`", "\\`")
    new_code = new_code.replace("${", "\\${")

    # New segment
    replacement = "{`" + new_code + "`}\n" + indent
    
    # Wait, `replace_range`...
    # Warning: `post` starts at `</SyntaxHighlighter>`.
    # We need to ensure we don't duplicate or lose the `}` closing the JSX expression?
    # In JSX: `<Component>{`string`}</Component>`
    # So we replaced content inside `{...}`.
    # The existing code is `{`...`}`.
    # My `open_brace_idx` points to `{` of `{`
    # My `close_tag_idx` points to `<` of `</SyntaxHighlighter>`
    
    # So range to replace is `open_brace_idx` to `close_tag_idx`.
    # We replace it with `{` + `new_code` + `}` (plus whitespace).
    
    # Let's clean up `new_code` (trim?) No, raw code.
    
    return content[:open_brace_idx] + "{`" + new_code + "`}\n" + indent + content[close_tag_idx:]

if __name__ == "__main__":
    update_dashboard()
