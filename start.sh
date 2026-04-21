#!/bin/bash
echo "Building FAISS index..."
python src/ingest.py
echo "Starting Streamlit..."
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0