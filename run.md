# Sentence Transformers Cosine Similarity Example

## Python Installation

Download Python 3.9.13 from the official archive:

```yaml
https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
```

### Run the installer

Check "Add Python 3.9 to PATH"

Click "Next" → "Install"

### Verify installation

```cmd
python --version
pip --version
```

## Create a Project Folder

```cmd
mkdir sentence-transformers-example
cd sentence-transformers-example
```

## Create a Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should now see (venv) at the beginning of your command prompt line.

## Upgrade pip

```cmd
python -m pip install --upgrade pip
```

## Install Required Libraries

```cmd
pip install sentence-transformers torch
```

## Save Your Python Script

Open Notepad or any text editor.

Paste this code:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["A cat sat on the mat.", "A cat uses mat to sit."]

embeddings = model.encode(sentences, convert_to_tensor=True)
similarity = util.cos_sim(embeddings[0], embeddings[1])

print(f"Cosine Similarity: {similarity.item():.4f}")
```

Save the file as: similarity.py in your project folder.

## Run the Script

In the Command Prompt (with virtual environment activated), run:

```cmd
python similarity.py
```

You should see output like:

```yaml
Cosine Similarity: 0.8792
```
