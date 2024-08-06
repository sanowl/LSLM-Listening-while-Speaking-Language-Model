
# LSLM: Listening-while-Speaking Language Model

## Overview

This project implements the Listening-while-Speaking Language Model (LSLM) as described in the paper "Language Model Can Listen While Speaking" by Ma et al. (2024). LSLM is an innovative approach to full duplex modeling in interactive speech language models, enabling real-time interaction and turn-taking in spoken dialogues.

## Key Features

- Full duplex modeling capability
- Real-time streaming for SSL encoder
- Token-based TTS generation
- Interruption handling
- Noise robustness
- Multi-fusion strategies (early, middle, late)
- Command-based and voice-based full duplex modeling

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers library
- Torchaudio
- Matplotlib
- NumPy

## Installation

```bash
git clone https://github.com/sanowl/LSLM-Listening-while-Speaking-Language-Model.git
```

## Usage

To train and evaluate the model:

```bash
python main.py
```

This script will:
1. Create and preprocess the dataset
2. Train the LSLM model
3. Evaluate on validation and test sets
4. Perform ablation studies
5. Conduct command-based and voice-based FDM tests
6. Analyze turn-taking performance
7. Generate sample speech output
8. Visualize attention weights and audio quantization

## Project Structure

- `main.py`: Main execution script
- `model/`: Contains model architecture components
- `utils/`: Utility functions for data processing and evaluation
- `data/`: Data loading and preprocessing scripts

## Citation

If you use this implementation in your research, please cite the original paper:

@article{ma2024language,
  title={Language Model Can Listen While Speaking},
  author={Ma, Ziyang and Song, Yakun and Du, Chenpeng and Cong, Jian and Chen, Zhuo and Wang, Yuping and Wang, Yuxuan and Chen, Xie},
  journal={arXiv preprint arXiv:2408.02622},
  year={2024}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to express our gratitude to the authors of the original paper for their innovative work in full duplex modeling for interactive speech language models.