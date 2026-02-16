# Extreme-LPSR
This repository implements license plate extreme super-resolution, published at **Neurocomputing**.

**Publication:**  
**Title:** *Embedding Similarity Guided License Plate Super Resolution*  
**Authors:** Abderrezzaq Sendjasni and  Mohamed-Chaker Larabi

**Read the article:** [https://www.sciencedirect.com/science/article/pii/S0925231225013293](https://www.sciencedirect.com/science/article/pii/S0925231225013293)

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

## How to cite
```
@article{SENDJASNI2025130657,
  title = {Embedding similarity guided license plate super resolution},
  journal = {Neurocomputing},
  volume = {651},
  pages = {130657},
  year = {2025},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2025.130657},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231225013293},
  author = {Abderrezzaq Sendjasni and Mohamed-Chaker Larabi}
}
```
