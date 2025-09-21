# medicortex.ai
This chatbot is designed to assist users in the healthcare domain by providing preliminary medical suggestions to patients seeking guidance online, both before and after consultations with healthcare professionals (HCPs). It is important to note that this chatbot is intended solely as a supportive tool to empower patients and should not be considered a substitute for professional medical advice, diagnosis, or treatment. The chatbot serves as a patient enabler, complementing, not replacing, the expertise and care provided by qualified healthcare professionals.

# Open in GitHub Codespaces

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repo=Indranil-Seal/medicortex.ai&ref=master)

[!NOTE]
When you open this project in GitHub Codespaces using the button above, the development environment is automatically set up for you. A virtual environment is created and all required Python packages from `requirements.txt` are installed automatically, so you can start coding right away without any manual setup.

# Environment Variables (.env file)

Create a `.env` file in the project root with the following format:

```env
# Path to the raw data CSV file
INPUT_LOAD = "/workspaces/medicortex.ai/data/rawdata.csv"

# Directory to store model objects
OUTPUT_DIR = "/workspaces/medicortex.ai/model_objects"

# Paths to FAISS index files
DOCTOR_INDEX_PATH = "/workspaces/medicortex.ai/model_objects/doctor_index.faiss"
PATIENT_INDEX_PATH = "/workspaces/medicortex.ai/model_objects/patient_index.faiss"
LOCATION_INDEX_PATH = "/workspaces/medicortex.ai/model_objects/desc_index.faiss"

# OpenAI API Key (replace with your own key)
OPENAI_API_KEY = "sk-..."
```

**Note:**
- All paths should be absolute or relative to the project root.
- Never share your OpenAI API key publicly.


# downloaddata.py
Use this python script to download the data from kagglehub onto your data folder on project repository. 

```bash
import kagglehub
import shutil
import os

# Download dataset (goes to kagglehub's cache)
path = kagglehub.dataset_download("yousefsaeedian/ai-medical-chatbot")
print("Downloaded to:", path)

# Your target directory
target_dir = r"enter/your/directory/"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move all files from download path to your target directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Files copied to:", target_dir)


## Set-up the Python Virtual Environment 
This project is built with **Python 3.10**.
To set up the Python virtual environment and install dependencies:

1. Ensure Python 3.10 is installed on your system.
2. Open your command prompt.
3. Navigate to your project directory.
4. Fetch the `requirements.txt` file from the repository:
    ```bash
    git pull origin main
    ```
5. Create a virtual environment:
    ```bash
    python -m venv env
    ```
6. Activate the virtual environment:
    ```bash
    env\Scripts\activate.bat
    ```
7. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
To activate the virtual environment, follow these steps:

1. Open your command prompt.
2. Navigate to your project directory.
3. Run the following command to activate the environment:

```bash
env\Scripts\activate.bat
```
Once activated, your prompt will show the environment name, and you can install dependencies locally.



# Git Ops & Guidlines

## GitOps:
1. Clone the ```main``` branch. 
``` bash
git clone https://github.com/Indranil-Seal/medicortex.ai.git
```
2. Create a ```develop/feature``` branch
```bash
git checkout main
git pull origin main
git checkout -b feature/<your-feature-name>
```
3. Work on your ```developments/changes/fixes```
``` bash
git add .
git commit -m "Add feature: short description"
```
4. Sync with Main Regularly
Ensure your branch stays up-to-date with the main branch to avoid conflicts.
```bash
git fetch origin
git rebase origin/main
# OR, if you prefer merging
# git merge origin/main
```
5. Push your branch to github
```bash
git push origin feature/<your-feature-name>
```
6. Open a Pull Request (PR)
```text
-Go to the repository on GitHub.
-Click "Compare & pull request" for your branch.
-Fill out the PR template with details of your changes.
-Assign reviewers and add relevant labels.
```
7. PR Review & Merge
```text
-Address reviewer comments and requested changes.
-Once approved, the PR will be merged by a maintainer.
-Delete your feature branch after merging to keep the repo clean.
```
8. Pull Latest Changes

After a successful merge, always pull the latest main before starting new work:
```bash
git checkout main
git pull origin main
```

## Repository Guidelines:
```text
1. main branch: Always contains production-ready code.
2. dev or develop branch: Serves as the integration branch for features before release.
3.Feature branches: Follow the naming convention feature/<feature-name>.
4. Hotfix branches: Use for urgent fixes, naming convention hotfix/<issue/description>.
5. Release branches: For preparing new production releases, naming convention release/<version>.