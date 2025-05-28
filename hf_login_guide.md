# 🤗 Hugging Face Login Guide

## Quick Login Steps

### On RunPod (where your model is):
```bash
# Method 1: Interactive login
huggingface-cli login

# Method 2: Direct token input
huggingface-cli login --token YOUR_TOKEN_HERE

# Method 3: Environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### Verify Login:
```bash
huggingface-cli whoami
```

## Token Permissions Needed:
- ✅ **Write access to contents of all repos you can access**
- ✅ **Read access to contents of all public gated repos you can access**
- ✅ **Manage your repos (create, delete, update)** *(recommended)*

## Security Tips:
- 🔒 Keep your token private
- 🔄 Rotate tokens periodically
- 🗑️ Delete unused tokens
- 📝 Use descriptive token names

## Test Upload:
```bash
# Once logged in, test the upload
python upload_to_huggingface.py --username YOUR_HF_USERNAME
``` 