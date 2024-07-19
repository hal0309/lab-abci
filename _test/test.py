import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import datetime


print(torch.cuda.is_available())