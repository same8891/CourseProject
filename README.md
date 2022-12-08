# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.


# Prerequisites

The following python packages are required for running our tool:

- streamlit (https://streamlit.io/)
- pytorch (https://pytorch.org/)
- transformers (https://huggingface.co/docs/transformers/index)
- simpletransformers (https://simpletransformers.ai/)
- scipy
- pandas

# Installation

First, git clone our repository:
```
git clone https://github.com/same8891/CourseProject.git
```

Then, go to the `code` folder:
```
cd CourseProject/code
```

Then, download these two Google Drive folders (our model parameters) to the current position, and rename them as `roberta` and `bert`, respectively.

- https://drive.google.com/drive/folders/1ESOczJViLGB7yyctyFch6C-3EhcOoq-J
- https://drive.google.com/drive/folders/10Qj5SHwn_jMGDFWzRs3nDx9sTtNZF3p9

Finally, launch the streamlit server:
```
streamlit run app.py
```

You should expect the following outputs in the terminal:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.186.103.205:8501
```
If the webpage is not automatically popping-up, please copy-paste the url to your browser and open.
