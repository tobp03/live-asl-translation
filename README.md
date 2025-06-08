# ✊ Real-time Sign Language Translation with Citizen ASL Dataset 

This project implements a full pipeline to translate sign language into text using pose estimation, deep learning, and a large language model (LLM) for sentence refinment. The live translation system is built using the [Citizen ASL Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/) and runs in real time with a webcam.

<p align="center">
  <a href="https://www.youtube.com/watch?v=gJ-PRa88E-M">
    <img src="demo/asldemo.gif" alt="Live ASL Translation Demo">
  </a>
</p>
Watch a real-time demo of sign language translation using our trained model and the Citizen ASL dataset.
> Disclaimer: I am an amateur signer, and the signing in this demo may not fully follow formal ASL grammar or conventions.

## Features
- Pose-based sign language recognition using MediaPipe Holistic
- Normalization and preprocessing of landmark sequences
- GRU-based model trained on 200 unique glosses (271 total classes including duplicates)
- Real-time webcam-based translation with LLM for sentence refinement (OpenAI GPT)


## Setup
### Install Dependencies
```bash
pip install -r requirements.txt
```
Alternatively, install dependencies manually ensuring the following versions:
- Python 3.10.17
- TensorFlow 2.19.0
- MediaPipe 0.10.9
- OpenAI 1.82.0

### Obtain API Key for LLM
This project uses [OpenAI GPT API](https://platform.openai.com) for sentence refinement during live translation. You need a valid API key with billing enabled. Crete a file named `.env` in the project root and add the following line:
```bash
API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Make sure you have `python-dotenv` installed.

## 🚀 Usage
- Full pipeline
   To train your own model and reproduce the full workflow : Open and run `main_livefeed.ipynb`. This notebook contains all steps end-to-end:
   1.  Download the [Citizen ASL Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/)
   2.  Place the dataset folder in the same directory as `main_livefeed.ipynb` (the project root)
   3.  Open and run the full pipeline step 1-3 on `main_livefeed.ipynb`
   4.  Finally, use the trained model for live translation
- Direct use with Pre-trained model
  If you prefer to skip training
  1. Ensure the pre-trained GRU model (`.keras`) is available in the `/models` folder (already provided)
  2. Open `main_livefeed.ipynb`
  3. Jump directly to **Step 4** to start live translation using the pre-trained model


## References
- **ASL Citizen Dataset**  
  Desai, A., Berger, L., Minakov, F. O., Milan, V., Singh, C., Pumphrey, K., Ladner, R. E., Daumé III, H., Lu, A. X., Caselli, N., & Bragg, D.  
  *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*.  
  arXiv preprint arXiv:2304.05934, 2023. [[Paper](https://arxiv.org/abs/2304.05934)]
- **Keypoint Preprocessing**  
  Roh, K., Lee, H., Hwang, E. J., Cho, S., & Park, J. C.  
  *Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition*.  
  In *LREC-COLING 2024 11th Workshop on the Representation and Processing of Sign Languages*, pp. 323–334. [[Paper](https://aclanthology.org/2024.signlang-1.36.pdf)]

## Related Publication
This project is related to the research paper:  
Purbojo, T., & Wijaya, A. (2023). *Pose-Based Sign Language Recognition Using Mediapipe Holistic and LSTM*.  
[ASSET Journal, Volume X, Issue Y, Pages A-B](https://journal2.upgris.ac.id/index.php/asset/article/view/1658)

