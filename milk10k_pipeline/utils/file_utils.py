"""
File I/O utilities for MILK10k pipeline
"""
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class FileManager:
    """Handles all file I/O operations for the pipeline"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = "") -> Path:
        """Save dictionary as JSON file"""
        save_dir = self.base_path / subdir if subdir else self.base_path
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def load_json(self, filename: str, subdir: str = "") -> Optional[Dict[str, Any]]:
        """Load JSON file as dictionary"""
        load_dir = self.base_path / subdir if subdir else self.base_path
        filepath = load_dir / f"{filename}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {filepath}: {e}")
            return None
    
    def save_csv(self, data: Union[pd.DataFrame, Dict, List], filename: str, subdir: str = "") -> Path:
        """Save data as CSV file"""
        save_dir = self.base_path / subdir if subdir else self.base_path
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{filename}.csv"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, dict):
            pd.DataFrame([data]).to_csv(filepath, index=False)
        elif isinstance(data, list):
            pd.DataFrame(data).to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported data type for CSV: {type(data)}")
        
        return filepath
    
    def load_csv(self, filename: str, subdir: str = "") -> Optional[pd.DataFrame]:
        """Load CSV file as DataFrame"""
        load_dir = self.base_path / subdir if subdir else self.base_path
        filepath = load_dir / f"{filename}.csv"
        
        if not filepath.exists():
            return None
        
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading CSV file {filepath}: {e}")
            return None
    
    def save_pickle(self, data: Any, filename: str, subdir: str = "") -> Path:
        """Save data as pickle file"""
        save_dir = self.base_path / subdir if subdir else self.base_path
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load_pickle(self, filename: str, subdir: str = "") -> Optional[Any]:
        """Load pickle file"""
        load_dir = self.base_path / subdir if subdir else self.base_path
        filepath = load_dir / f"{filename}.pkl"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {filepath}: {e}")
            return None
    
    def save_numpy(self, array: np.ndarray, filename: str, subdir: str = "") -> Path:
        """Save numpy array"""
        save_dir = self.base_path / subdir if subdir else self.base_path
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{filename}.npy"
        np.save(filepath, array)
        
        return filepath
    
    def load_numpy(self, filename: str, subdir: str = "") -> Optional[np.ndarray]:
        """Load numpy array"""
        load_dir = self.base_path / subdir if subdir else self.base_path
        filepath = load_dir / f"{filename}.npy"
        
        if not filepath.exists():
            return None
        
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Error loading numpy file {filepath}: {e}")
            return None
    
    def list_files(self, subdir: str = "", pattern: str = "*") -> List[Path]:
        """List files in directory"""
        search_dir = self.base_path / subdir if subdir else self.base_path
        
        if not search_dir.exists():
            return []
        
        return list(search_dir.glob(pattern))
    
    def create_directory_structure(self, structure: Dict[str, Any]):
        """Create directory structure from nested dictionary"""
        def create_dirs(base_path: Path, struct: Dict[str, Any]):
            for key, value in struct.items():
                dir_path = base_path / key
                dir_path.mkdir(exist_ok=True)
                
                if isinstance(value, dict):
                    create_dirs(dir_path, value)
        
        create_dirs(self.base_path, structure)
    
    def get_file_info(self, filepath: Path) -> Dict[str, Any]:
        """Get information about a file"""
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime,
            'is_file': filepath.is_file(),
            'is_directory': filepath.is_dir(),
            'extension': filepath.suffix,
            'name': filepath.name,
            'stem': filepath.stem
        }

# ==================== SPECIALIZED UTILITIES ====================

def save_processing_results(results: List[Dict], output_path: Path, 
                          filename_prefix: str = "results") -> Dict[str, Path]:
    """Save processing results in multiple formats"""
    file_manager = FileManager(output_path)
    saved_files = {}
    
    # Save as CSV
    csv_path = file_manager.save_csv(results, f"{filename_prefix}_detailed", "reports")
    saved_files['csv'] = csv_path
    
    # Save as JSON
    json_path = file_manager.save_json(results, f"{filename_prefix}_detailed", "reports")
    saved_files['json'] = json_path
    
    # Save as pickle for full Python object preservation
    pickle_path = file_manager.save_pickle(results, f"{filename_prefix}_detailed", "reports")
    saved_files['pickle'] = pickle_path
    
    return saved_files

def load_processing_results(output_path: Path, filename_prefix: str = "results", 
                          format_preference: str = "csv") -> Optional[List[Dict]]:
    """Load processing results with format preference"""
    file_manager = FileManager(output_path)
    
    # Try preferred format first
    if format_preference == "csv":
        df = file_manager.load_csv(f"{filename_prefix}_detailed", "reports")
        if df is not None:
            return df.to_dict('records')
    
    elif format_preference == "json":
        data = file_manager.load_json(f"{filename_prefix}_detailed", "reports")
        if data is not None:
            return data
    
    elif format_preference == "pickle":
        data = file_manager.load_pickle(f"{filename_prefix}_detailed", "reports")
        if data is not None:
            return data
    
    # Try other formats as fallbacks
    formats_to_try = [("pickle", "load_pickle"), ("json", "load_json"), ("csv", "load_csv")]
    
    for fmt, method_name in formats_to_try:
        if fmt != format_preference:
            try:
                method = getattr(file_manager, method_name)
                data = method(f"{filename_prefix}_detailed", "reports")
                
                if data is not None:
                    if fmt == "csv" and isinstance(data, pd.DataFrame):
                        return data.to_dict('records')
                    else:
                        return data
            except:
                continue
    
    return None

def create_backup(source_path: Path, backup_dir: Path, backup_name: str = None) -> Path:
    """Create backup of files or directories"""
    import shutil
    from datetime import datetime
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_backup_{timestamp}"
    
    backup_path = backup_dir / backup_name
    
    if source_path.is_file():
        shutil.copy2(source_path, backup_path)
    elif source_path.is_dir():
        shutil.copytree(source_path, backup_path)
    else:
        raise ValueError(f"Source path does not exist: {source_path}")
    
    return backup_path

def cleanup_temporary_files(temp_dir: Path, max_age_hours: float = 24):
    """Clean up temporary files older than specified age"""
    import time
    
    if not temp_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in temp_dir.rglob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    print(f"Deleted old temporary file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

def compress_large_files(directory: Path, size_threshold_mb: float = 100, 
                        compression_format: str = "gzip"):
    """Compress large files in directory"""
    import gzip
    import zipfile
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if size_mb > size_threshold_mb:
                if compression_format == "gzip":
                    compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original file
                    file_path.unlink()
                    print(f"Compressed {file_path} -> {compressed_path}")
                
                elif compression_format == "zip":
                    compressed_path = file_path.with_suffix(".zip")
                    with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(file_path, file_path.name)
                    
                    # Remove original file
                    file_path.unlink()
                    print(f"Compressed {file_path} -> {compressed_path}")

def validate_dataset_structure(dataset_path: Path, expected_structure: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset directory structure"""
    results = {
        'valid': True,
        'missing_directories': [],
        'missing_files': [],
        'extra_items': [],
        'issues': []
    }
    
    def check_structure(current_path: Path, structure: Dict[str, Any], level: int = 0):
        for item_name, item_spec in structure.items():
            item_path = current_path / item_name
            
            if not item_path.exists():
                if isinstance(item_spec, dict) and item_spec.get('required', True):
                    if item_spec.get('type') == 'directory':
                        results['missing_directories'].append(str(item_path))
                    else:
                        results['missing_files'].append(str(item_path))
                    results['valid'] = False
            
            elif item_path.is_dir() and isinstance(item_spec, dict):
                if 'contents' in item_spec:
                    check_structure(item_path, item_spec['contents'], level + 1)
    
    check_structure(dataset_path, expected_structure)
    return results

# ==================== CONFIGURATION MANAGEMENT ====================

class ConfigManager:
    """Manage configuration files and settings"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], name: str):
        """Save configuration"""
        config_path = self.config_dir / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load configuration"""
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        return [f.stem for f in self.config_dir.glob("*.json")]
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """Merge multiple configurations"""
        merged = {}
        
        for name in config_names:
            config = self.load_config(name)
            if config:
                merged.update(config)
        
        return merged