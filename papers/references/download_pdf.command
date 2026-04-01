#!/bin/bash
cd "$(dirname "$0")"
echo "Downloading PDF from arXiv..."
curl -L -o gleave_mce_irl_primer_2022.pdf "https://arxiv.org/pdf/2203.11409"
echo "Download complete!"
echo "File saved to: $(pwd)/gleave_mce_irl_primer_2022.pdf"
ls -la gleave_mce_irl_primer_2022.pdf
sleep 3
