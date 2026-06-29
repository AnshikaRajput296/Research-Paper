import os
import tempfile


UPLOAD_DIR = "uploaded_papers"


def save_uploaded_files(uploaded_files) -> list[str]:
    """Save Streamlit uploaded files to disk and return their paths."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paths = []

    for f in uploaded_files:
        dest = os.path.join(UPLOAD_DIR, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        paths.append(dest)

    return paths
