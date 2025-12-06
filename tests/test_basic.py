"""
Basic unit tests that don't require downloading models.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_model(self):
        """Test importing model module."""
        try:
            import model
            self.assertTrue(hasattr(model, 'MultiModalReviewAnalyzer'))
            self.assertTrue(hasattr(model, 'create_model'))
        except ImportError as e:
            self.fail(f"Failed to import model: {e}")
    
    def test_import_data_loader(self):
        """Test importing data_loader module."""
        try:
            import data_loader
            self.assertTrue(hasattr(data_loader, 'AmazonReviewDataset'))
            self.assertTrue(hasattr(data_loader, 'create_data_loaders'))
        except ImportError as e:
            self.fail(f"Failed to import data_loader: {e}")
    
    def test_import_train(self):
        """Test importing train module."""
        try:
            import train
            self.assertTrue(hasattr(train, 'Trainer'))
        except ImportError as e:
            self.fail(f"Failed to import train: {e}")
    
    def test_import_evaluate(self):
        """Test importing evaluate module."""
        try:
            import evaluate
            self.assertTrue(hasattr(evaluate, 'Evaluator'))
            self.assertTrue(hasattr(evaluate, 'Predictor'))
        except ImportError as e:
            self.fail(f"Failed to import evaluate: {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration and basic setup."""
    
    def test_config_file_exists(self):
        """Test that config file exists."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.assertTrue(os.path.exists(config_path))
    
    def test_requirements_file_exists(self):
        """Test that requirements file exists."""
        req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        self.assertTrue(os.path.exists(req_path))
    
    def test_readme_exists(self):
        """Test that README exists."""
        readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        self.assertTrue(os.path.exists(readme_path))


class TestProjectStructure(unittest.TestCase):
    """Test that project structure is correct."""
    
    def test_src_directory_exists(self):
        """Test src directory exists."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        self.assertTrue(os.path.isdir(src_path))
    
    def test_scripts_directory_exists(self):
        """Test scripts directory exists."""
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        self.assertTrue(os.path.isdir(scripts_path))
    
    def test_notebooks_directory_exists(self):
        """Test notebooks directory exists."""
        notebooks_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
        self.assertTrue(os.path.isdir(notebooks_path))
    
    def test_essential_files_exist(self):
        """Test that essential files exist."""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        
        essential_files = [
            'src/model.py',
            'src/data_loader.py',
            'src/train.py',
            'src/evaluate.py',
            'src/__init__.py',
            'scripts/prepare_data.py',
            'requirements.txt',
            'README.md',
            'LICENSE'
        ]
        
        for file in essential_files:
            file_path = os.path.join(base_path, file)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file}")


if __name__ == '__main__':
    unittest.main()
