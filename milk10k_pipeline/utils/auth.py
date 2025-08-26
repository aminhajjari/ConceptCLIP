"""
Authentication utilities for MILK10k pipeline
"""
import os
from huggingface_hub import login

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    print("Setting up Hugging Face authentication...")
    
    # Method 1: Environment variable
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("Found HF_TOKEN in environment variables")
        try:
            login(token=hf_token, new_session=False)
            print("Hugging Face authentication successful via token")
            return True
        except Exception as e:
            print(f"Token authentication failed: {e}")
    
    # Method 2: Interactive login
    try:
        login(new_session=False)
        print("Hugging Face login successful")
        return True
    except Exception as e:
        print(f"Interactive login failed: {e}")
    
    # Method 3: Instructions for manual setup
    print("\n" + "="*50)
    print("HUGGING FACE AUTHENTICATION REQUIRED")
    print("="*50)
    print("Please set up authentication using one of these methods:")
    print("1. Set HF_TOKEN environment variable:")
    print("   export HF_TOKEN=your_token_here")
    print("2. Run: huggingface-cli login")
    print("3. Use notebook_login() in Jupyter notebooks")
    print("="*50)
    
    return False

def verify_authentication():
    """Verify that Hugging Face authentication is working"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"Authentication verification failed: {e}")
        return False