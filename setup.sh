# Exit immediately if a command exits with a non-zero status
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt