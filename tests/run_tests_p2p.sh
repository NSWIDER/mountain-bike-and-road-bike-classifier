#!/bin/bash
set -euo pipefail
# Exit immediately if any command exits with a non-zero status
set -e

echo "========================================="
echo "Starting Python Unit Tests..."
echo "========================================="

# Option A: Run a single explicit test file
python3 -m unittest -f -v ./golden_tests/p2p.py  

# Option B: Or discover and run all tests matching 'test_*.py'
# python3 -m unittest discover -v -s . -p "test_*.py"

echo "========================================="
echo "All tests completed successfully!"
echo "========================================="
