# This is the official repository of the paper:
#### [IDDR-NGP:Incorporating Detectors for Distractors Removal with Instant Neural Radiance Field](https://dl.acm.org/doi/pdf/10.1145/3581783.3612045)


# Install
### Install with pip
```bash
pip install -r requirements.txt
```

### Install with conda
```bash
conda env create -f environment.yml
conda activate iddr-ngp
```

# Data 

We use the same data format as instant-ngp. We provide three real scenes mentioned in the paper, please [download](https://drive.google.com/drive/folders/1DyvVdxdTYa_2920bq5HrYLDHb3IfKQ3A?usp=drive_link)  and put them under `./data`.

# Usage
```bash
# use the 'outdoor' dataset
python main_nerf.py data/outdoor --workspace outdoor_test -O --demask
```

# Result
The processed image will be saved in the `./your_workplace/result/`.

# Citation

If you find this work useful, a citation will be appreciated via:
```
@inproceedings{huang2023iddr,
  title={IDDR-NGP: Incorporating Detectors for Distractors Removal with Instant Neural Radiance Field},
  author={Huang, Xianliang and Gou, Jiajie and Chen, Shuhang and Zhong, Zhizhou and Guan, Jihong and Zhou, Shuigeng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1343--1351},
  year={2023}
}
```

# Acknowledgement
The implementation is based on the [torch-ngp](https://github.com/ashawkey/torch-ngp).

# Contact
If you have any question, please feel free to contact **huangxl21@m.fudan.edu.cn**.
