# ✊ Real-time Sign Language Translation with Citizen ASL Dataset 🤚

This project implements a full pipeline to translate sign language into text using pose estimation, deep learning, and a large language model (LLM) for sentence refinment. The live translation system is built using the [Citizen ASL Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/) and runs in real time with a webcam.

[![Live Sign Language Translation Demo](https://img.youtube.com/vi/gJ-PRa88E-M/maxresdefault.jpg)](https://www.youtube.com/watch?v=gJ-PRa88E-M)
Watch a real-time demo of sign language translation using our trained model and the Citizen ASL dataset.
> Disclaimer: I am an amateur signer, and the signing in this demo may not fully follow formal ASL grammar or conventions.

## Features
- Pose-based sign language recognition using MediaPipe Holistic
- Normalization and preprocessing of landmark sequences
- GRU-based model trained on 200 unique glosses (271 total classes including duplicates)
- Real-time webcam-based translation with LLM for sentence refinement (OpenAI GPT)
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

