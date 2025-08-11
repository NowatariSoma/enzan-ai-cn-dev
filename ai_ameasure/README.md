# AI Displacement Analysis

This project provides a web application for analyzing displacement data using Streamlit. The application allows users to visualize and predict displacement and settlement based on excavation data.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run gui_displacement_temporal_spacial_analysis.py
   ```

2. **prepare dataset and config.json**

config.json
```json
{
    "selected_index": 0,
    "selected_folder": "",
    "input_folder": "path/to/data_folder"
}
```

3. **Build the Docker image:**

   ```bash
   docker build -t ai-measure .
   ```

4. **Run the Docker container:**

   ```bash
   docker run -p 8501:8501 ai-measure
   ```

5. **Access the application:**

   Open your web browser and go to `http://localhost:8501` to access the displacement analysis application.

6. sample data

put unzipped folders to `path/to/data_folder`

[sampledata.zip](https://github.com/user-attachments/files/21032134/sampledata.zip)

