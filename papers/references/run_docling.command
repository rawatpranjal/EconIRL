#!/bin/bash
cd "$(dirname "$0")"
PDF="gleave_mce_irl_primer_2022.pdf"
OUTPUT_DIR="$(pwd)"

echo "Running docling on $PDF..."
echo ""

# Try docling CLI first
if command -v docling &> /dev/null; then
    echo "Found docling CLI, running..."
    docling "$PDF" --output "$OUTPUT_DIR"
    if [ $? -eq 0 ]; then
        echo ""
        echo "Success! Checking for output..."
        ls -la *.md 2>/dev/null
        echo ""
        echo "Done!"
        sleep 5
        exit 0
    else
        echo "docling CLI failed, trying Python API..."
    fi
else
    echo "docling CLI not found, trying Python API..."
fi

# Try Python API
python3 -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('$PDF')
md = result.document.export_to_markdown()
with open('gleave_mce_irl_primer_2022.md', 'w') as f:
    f.write(md)
print('Markdown saved successfully!')
print(f'Output length: {len(md)} characters')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "Success!"
    ls -la gleave_mce_irl_primer_2022.md
else
    echo ""
    echo "Python API also failed. You may need to install docling:"
    echo "  pip install docling"
fi

echo ""
sleep 5
