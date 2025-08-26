#!/usr/bin/env python3
"""
Quick download ConceptCLIP - Secure Version
Token is provided via environment variable or argument
"""
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import login, snapshot_download
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "JerrryNie/ConceptCLIP"
CACHE_DIR = "./models/conceptclip"

def get_token():
    """Get HuggingFace token from environment or prompt"""
    
    # Try environment variable first
    token = os.environ.get('HF_TOKEN')
    if token:
        print("✅ Found HF_TOKEN in environment")
        return token
    
    # Try common environment variable names
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        print("✅ Found HUGGINGFACE_TOKEN in environment")
        return token
    
    # If no environment token, this will be set by Naval script
    print("⚠️  No HF_TOKEN found in environment")
    print("💡 Token should be provided via environment variable or Naval script")
    return None

def quick_download(token=None):
    """Quick download ConceptCLIP"""
    
    if not token:
        token = get_token()
    
    if not token:
        print("❌ No HuggingFace token available!")
        print("💡 Set token via: export HF_TOKEN=your_token_here")
        return False
    
    print("🚀 Quick ConceptCLIP Download")
    print("="*50)
    print(f"🔑 Using token: {token[:12]}...")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"🤗 Model: {MODEL_NAME}")
    
    # Create cache directory
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    transformers_cache = cache_path / "transformers"
    
    try:
        # Set token and login
        os.environ['HF_TOKEN'] = token
        login(token=token, new_session=False)
        print("✅ Authentication successful")
        
        # Download model
        print("\n📥 Downloading ConceptCLIP model...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(transformers_cache),
            trust_remote_code=True,
            torch_dtype="auto",
            token=token
        )
        print("✅ Model downloaded and cached")
        
        # Download processor
        print("\n📥 Downloading ConceptCLIP processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=str(transformers_cache),
            trust_remote_code=True,
            token=token
        )
        print("✅ Processor downloaded and cached")
        
        # Create usage script
        create_usage_script(cache_path, transformers_cache)
        
        # Create classifier update
        create_classifier_update(cache_path, transformers_cache)
        
        print("\n" + "="*60)
        print("🎉 CONCEPTCLIP DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"📁 Cached in: {cache_path.absolute()}")
        print(f"🧪 Test script: {cache_path / 'test_conceptclip.py'}")
        print(f"🔧 Classifier update: {cache_path / 'update_classifier.py'}")
        
        print("\n✅ Next steps:")
        print("1. Test: python models/conceptclip/test_conceptclip.py")
        print("2. Update your classifier: python models/conceptclip/update_classifier.py")
        print("3. Run your pipeline normally (no token needed!)")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def create_usage_script(cache_path: Path, transformers_cache: Path):
    """Create test script"""
    
    test_script = f'''#!/usr/bin/env python3
"""Test cached ConceptCLIP"""
import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
from PIL import Image

def test_conceptclip():
    print("🧪 Testing cached ConceptCLIP...")
    
    try:
        # Load from cache (no token needed)
        model = AutoModel.from_pretrained(
            "{MODEL_NAME}",
            cache_dir="{transformers_cache}",
            trust_remote_code=True,
            torch_dtype="auto",
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(
            "{MODEL_NAME}",
            cache_dir="{transformers_cache}",
            trust_remote_code=True,
            local_files_only=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"✅ Model loaded successfully on {{device}}")
        
        # Test with dummy data
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_texts = ["normal tissue", "abnormal pathology"]
        
        inputs = processor(
            images=dummy_image,
            text=test_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs['logits_per_image'], dim=-1)
        
        print("✅ Test successful!")
        print(f"📊 Test probabilities: {{[f'{{p:.3f}}' for p in probs[0].cpu().tolist()]}}")
        
    except Exception as e:
        print(f"❌ Test failed: {{e}}")

if __name__ == "__main__":
    test_conceptclip()
'''
    
    with open(cache_path / "test_conceptclip.py", 'w') as f:
        f.write(test_script)

def create_classifier_update(cache_path: Path, transformers_cache: Path):
    """Create classifier update script"""
    
    update_script = f'''#!/usr/bin/env python3
"""Update your classifier to use cached ConceptCLIP"""
import shutil
from pathlib import Path

def update_classifier():
    print("🔧 Updating ConceptCLIP classifier...")
    
    # Path to your classifier
    classifier_path = Path("classification/conceptclip_classifier.py")
    
    if not classifier_path.exists():
        print(f"❌ Classifier not found: {{classifier_path}}")
        print("💡 Make sure you're running from your project root")
        return
    
    print("💡 Manual update instructions:")
    print("="*50)
    print("Replace your _load_models method with:")
    print()
    print('''def _load_models(self):
    """Load ConceptCLIP from cache (no token needed)"""
    try:
        # Load model from cache
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir="{transformers_cache}",
            trust_remote_code=True,
            torch_dtype="auto",
            local_files_only=True
        )
        
        # Load processor from cache
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir="{transformers_cache}",
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.model = self.model.to(self.device)
        print(f"✅ ConceptCLIP loaded from cache")
        
    except Exception as e:
        print(f"❌ Error loading cached ConceptCLIP: {{e}}")
        raise e''')

if __name__ == "__main__":
    update_classifier()
'''
    
    with open(cache_path / "update_classifier.py", 'w') as f:
        f.write(update_script)

def main():
    parser = argparse.ArgumentParser(description="Download ConceptCLIP (secure version)")
    parser.add_argument('--token', type=str, help='HuggingFace token')
    
    args = parser.parse_args()
    
    success = quick_download(args.token)
    if success:
        print("\n🎉 Ready to use ConceptCLIP!")
    else:
        print("\n❌ Download failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()