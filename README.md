# Extreme-LPSR
This repository implements license plate extreme super-resolution, currently under review.

**Publication:**  
**Title:** *Embedding Similarity Guided License Plate Super Resolution*  
**Authors:** Abderrezzaq Sendjasni and  Mohamed-Chaker Larabi

**Neurocomputing:** [https://www.sciencedirect.com/science/article/pii/S0925231225013293](https://www.sciencedirect.com/science/article/pii/S0925231225013293
**ArXiv:** [arxiv.org/abs/2501.01483v2](https://arxiv.org/abs/2501.01483v2)

---
## Installation
```
git clone https://github.com/sendjasni/Extreme-LPSR.git
cd Extreme-LPSR
pip install -r requirements.txt
```
## Requirements
```
torch>=1.10
torchvision
numpy
```
## Evaluation

To evaluate the model on a sample low-resolution image:
- Place your low-resolution image at `samples/lr.png`
- Place the corresponding high-resolution ground truth at `samples/hr.png`
- Download the pretrained model weights and place them in `checkpoints`

## Project structure
```
Extreme-LPSR/
├── model.py                        
├── eval.py              
├── README.md
└── requirements.txt
```
## Contact

For questions or collaborations, open an issue or contact via email: abderrezzaq.sendjasni@univ-poitiers.fr

## License
This project is licensed under the MIT License.
