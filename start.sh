#!/bin/bash

# GNCIPL Project 6 - Render Startup Script
echo "🚀 Starting GNCIPL Project 6..."

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit

# Create config.toml
cat > ~/.streamlit/config.toml << EOF
[general]
email = ""

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = $PORT
address = "0.0.0.0"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF

# Start the application
echo "🌐 Starting Streamlit on port $PORT..."
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false