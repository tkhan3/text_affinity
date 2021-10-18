#!/bin/bash
sh /opt/conda/etc/profile.d/conda.sh
conda activate text_affinity
uvicorn webapp.text_affinity:app --reload