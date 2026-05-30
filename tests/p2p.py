#!/usr/bin/env bash

import unittest
import subprocess
import sys
from pathlib import Path
class TestScripts(unittest.TestCase):
    def test_train_script_execution(self):
        
        parent_dir = Path(__file__).resolve().parent.parent

        # 2. Construct the full path to the target script
        target_script = parent_dir / "train.py"
        # Run the script using the current Python interpreter
        process = subprocess.run(
            [sys.executable, target_script],
            capture_output=True,
            text=True,
            check=True
        )
    

        # Assertions
        assert process.returncode == 0
        target_dir = Path(parent_dir / "logs_train/train")
        
        # Check if any item in the directory is a file
        file_exists = any(item.is_file() for item in target_dir.iterdir())
        
        # Assert the condition is True
        self.assertTrue(file_exists, f"No files found in {target_dir}")
       
    def test_test_script_execution(self):
        parent_dir = Path(__file__).resolve().parent.parent

        # 2. Construct the full path to the target script
        target_script = parent_dir / "test.py"
        # Run the script using the current Python interpreter
        process = subprocess.run(
            [sys.executable, target_script],
            capture_output=True,
            text=True,
            check=True
        )

        # Assertions
        assert process.returncode == 0
        #assert "The result is 30" in process.stdout
        target_dir = Path(parent_dir / "logs_test")
        
        # Check if any item in the directory is a file
        file_exists = any(item.is_file() for item in target_dir.iterdir())
        
        # Assert the condition is True
        self.assertTrue(file_exists, f"No files found in {target_dir}")

if __name__ == '__main__':
    unittest.main()
